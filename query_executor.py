"""
Execute SQL queries safely and return a pandas DataFrame.

Only read-only queries are allowed (single SELECT or WITH ... SELECT).
Includes optional column validation (sqlglot) and AI-assisted fix + retry.
"""

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import sqlglot
from sqlglot import exp
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from schema_utils import resolve_table_name


# Words that must not start the main query (single-statement check).
_DISALLOWED_STARTS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "truncate",
    "create",
    "grant",
    "revoke",
    "merge",
    "replace",
    "call",
    "execute",
)


def _normalize_for_check(sql: str) -> str:
    """Trim whitespace and a trailing semicolon for validation."""
    s = sql.strip()
    if s.endswith(";"):
        s = s[:-1].strip()
    return s


def validate_read_only_sql(sql_query: str) -> Tuple[bool, str]:
    """
    Check that sql_query is a single read-only statement.

    Returns (True, "") if OK, else (False, human-readable reason).
    """
    if not sql_query or not sql_query.strip():
        return False, "SQL query is empty. Paste or generate a SELECT query."

    normalized = _normalize_for_check(sql_query)

    parts = [p.strip() for p in normalized.split(";") if p.strip()]
    if len(parts) > 1:
        return (
            False,
            "Multiple statements are not allowed. Run one SELECT (or WITH ... SELECT) at a time.",
        )

    body = parts[0] if parts else normalized
    lower = body.lower().strip()

    if not (lower.startswith("select") or lower.startswith("with")):
        first_word = lower.split()[0] if lower.split() else ""
        if first_word in _DISALLOWED_STARTS:
            return (
                False,
                f"This tool only runs safe read queries. `{first_word.upper()}` is not allowed.",
            )
        return (
            False,
            "Only SELECT queries are allowed (you may start with WITH for a CTE). "
            "Examples: `SELECT ...` or `WITH ... AS (...) SELECT ...`.",
        )

    first_token = re.match(r"^\s*(\w+)", body, re.IGNORECASE)
    if first_token:
        fw = first_token.group(1).lower()
        if fw in _DISALLOWED_STARTS:
            return False, f"This tool only runs SELECT-style reads. `{fw.upper()}` is not allowed."

    return True, ""


def _is_select_query(sql_query: str) -> bool:
    """Return True if validation passes (backward-compatible helper)."""
    ok, _ = validate_read_only_sql(sql_query)
    return ok


def validate_sql_against_schema(
    sql: str,
    schema_details: Dict[str, List[dict]],
    dialect: str = "mysql",
) -> Tuple[bool, List[str]]:
    """
    Parse SQL and check that qualified columns exist on the resolved table.

    Uses sqlglot (best-effort). If parsing fails, returns (True, []) and lets the DB decide.
    Unqualified columns are only checked when exactly one real table appears in FROM/JOIN.
    """
    issues: List[str] = []
    if not schema_details:
        return True, []

    read = "postgres" if dialect.lower() in ("postgresql", "postgres") else "mysql"

    try:
        parsed = sqlglot.parse_one(sql, read=read)
    except Exception:
        return True, []

    # Map alias -> real table name (sqlglot gives t.name and t.alias as strings).
    alias_to_table: Dict[str, str] = {}
    for t in parsed.find_all(exp.Table):
        real = str(t.name).strip('"').strip("`")
        alias_to_table[real] = real
        if t.alias:
            al = str(t.alias).strip('"').strip("`")
            alias_to_table[al] = real

    unique_real_tables = set(alias_to_table.values())

    for col in parsed.find_all(exp.Column):
        if isinstance(col.this, exp.Star):
            continue
        cname = str(col.name).strip('"').strip("`")
        tbl_part = col.table
        tbl = str(tbl_part).strip('"').strip("`") if tbl_part else None

        real_table: Optional[str] = None
        if tbl:
            base = alias_to_table.get(tbl) or alias_to_table.get(tbl.lower())
            if not base:
                issues.append(
                    f"Unknown table or alias `{tbl}` for column `{cname}`."
                )
                continue
            real_table = resolve_table_name(schema_details, base)
            if not real_table:
                issues.append(f"Table `{base}` is not in the schema.")
                continue
        else:
            if len(unique_real_tables) == 1:
                only = next(iter(unique_real_tables))
                real_table = resolve_table_name(schema_details, only)
            else:
                continue

        if not real_table or real_table not in schema_details:
            continue

        col_names_lower = {
            c["name"].lower(): c["name"] for c in schema_details[real_table]
        }
        if cname.lower() not in col_names_lower:
            available = [c["name"] for c in schema_details[real_table]]
            issues.append(
                f"The column `{cname}` does not exist in table `{real_table}`. "
                f"Available columns: {', '.join(available)}."
            )

    return len(issues) == 0, issues


def execute_sql_query(engine: Engine, sql_query: str) -> pd.DataFrame:
    """
    Run a safe read-only query and return rows as a DataFrame.

    Raises:
        ValueError: empty or invalid (non-SELECT) SQL
        RuntimeError: database / connection / SQL execution error
    """
    if engine is None:
        raise ValueError("Database is not connected. Use “Connect and Detect Schema” first.")

    ok, reason = validate_read_only_sql(sql_query)
    if not ok:
        raise ValueError(f"Invalid or unsafe SQL: {reason}")

    to_run = _normalize_for_check(sql_query)

    try:
        with engine.connect() as connection:
            result = connection.execute(text(to_run))
            rows = result.fetchall()
            columns = list(result.keys())
        return pd.DataFrame(rows, columns=columns)
    except SQLAlchemyError as exc:
        raise RuntimeError(
            f"Query execution failed (check table/column names and SQL syntax): {exc}"
        ) from exc


def execute_sql_query_with_fixes(
    engine: Engine,
    sql_query: str,
    schema_text: str,
    schema_details: Optional[Dict[str, List[dict]]],
    question: str = "",
    sql_dialect: str = "mysql",
    max_llm_fixes: int = 3,
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Validate, optionally fix column issues via LLM, run query, retry on DB errors.

    Returns:
        (dataframe, final_sql_used, user_messages)

    user_messages are short strings you can show with st.info (e.g. auto-correct notices).
    """
    # Lazy import avoids circular dependency at module load time.
    from llm_sql_generator import fix_sql_using_schema_and_error, intent_mismatch_feedback  # noqa: PLC0415

    messages: List[str] = []
    sql = (sql_query or "").strip()
    fixes_used = 0

    if question:
        mismatch = intent_mismatch_feedback(question, sql)
        if mismatch:
            messages.append("SQL intent mismatch detected. Regenerating accurate SQL...")
            try:
                sql = fix_sql_using_schema_and_error(
                    broken_sql=sql,
                    schema_text=schema_text,
                    question=question,
                    error_message=f"INTENT MISMATCH:\n{mismatch}",
                )
                fixes_used += 1
            except Exception:
                pass

    while True:
        ok, reason = validate_read_only_sql(sql)
        if not ok:
            raise ValueError(f"Invalid or unsafe SQL: {reason}")

        if schema_details:
            col_ok, col_issues = validate_sql_against_schema(
                sql, schema_details, dialect=sql_dialect
            )
            if not col_ok and fixes_used < max_llm_fixes:
                messages.append(
                    "Column mismatch detected. Trying to auto-correct the query with AI…"
                )
                hint = "\n".join(col_issues)
                sql = fix_sql_using_schema_and_error(
                    broken_sql=sql,
                    schema_text=schema_text,
                    question=question,
                    error_message=hint,
                )
                fixes_used += 1
                continue
            if not col_ok:
                raise ValueError(
                    "SQL still references columns not in your schema after auto-fix attempts:\n"
                    + "\n".join(col_issues)
                )

        try:
            df = execute_sql_query(engine, sql)
            return df, sql, messages
        except RuntimeError as exc:
            err_text = str(exc)
            if fixes_used >= max_llm_fixes:
                raise RuntimeError(err_text) from exc
            messages.append(
                "The database reported an error. Asking AI to fix the SQL based on that error…"
            )
            sql = fix_sql_using_schema_and_error(
                broken_sql=sql,
                schema_text=schema_text,
                question=question,
                error_message=err_text,
            )
            fixes_used += 1
