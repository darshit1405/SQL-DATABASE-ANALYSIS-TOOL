"""
Helpers to inspect database schema.

Builds clear, structured text for the LLM and supports validation helpers.
"""

import re
from typing import Dict, List, Optional, Tuple

from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


# Simple fuzzy hints: user / LLM wording → likely real column names (schema must still contain them).
FUZZY_COLUMN_HINTS: Dict[str, List[str]] = {
    "sales": ["amount", "total_amount", "total", "order_total", "price", "revenue"],
    "sale": ["amount", "total_amount", "total"],
    "amount": ["total_amount", "amount", "order_total"],
    "total": ["total_amount", "amount", "order_total"],
    "date": ["order_date", "created_at", "updated_at"],
    "customer": ["customer_id", "name", "customer_name", "full_name"],
    "order": ["order_id", "id"],
}


def _is_likely_numeric_aggregate_type(type_str: str) -> bool:
    """Guess if a column type is suitable for SUM / AVG (heuristic)."""
    u = str(type_str).upper()
    if re.search(r"\bDATE\b", u) and "DATETIME" not in u and "TIMESTAMP" not in u:
        return False
    if "BOOL" in u:
        return False
    keywords = (
        "INT",
        "DECIMAL",
        "NUMERIC",
        "FLOAT",
        "DOUBLE",
        "REAL",
        "MONEY",
        "SMALLINT",
        "BIGINT",
        "TINYINT",
    )
    return any(k in u for k in keywords)


def _column_name_suggests_money(col_name: str) -> bool:
    """
    Heuristic: column name looks like a monetary / quantity-money field.

    Prefer names containing amount, price, sales, revenue, cost, or clear *total* patterns.
    Skip obvious id columns.
    """
    n = col_name.lower().strip('"`')
    if n.endswith("_id") and "amount" not in n:
        return False

    money_tokens = ("amount", "price", "sales", "revenue", "cost", "payment", "subtotal")
    if any(t in n for t in money_tokens):
        return True
    if n == "total" or n.startswith("total_") or n.endswith("_total"):
        return True
    return False


def list_monetary_columns(
    schema_details: Dict[str, List[dict]],
) -> List[Tuple[str, str]]:
    """
    Return [(table_name, column_name), ...] for columns that look monetary and are numeric.

    Used in the LLM schema block so the model picks SUM(...) targets correctly.
    """
    out: List[Tuple[str, str]] = []
    for table_name in sorted(schema_details.keys()):
        for col in schema_details[table_name]:
            cname = col.get("name")
            if not cname:
                continue
            if not _column_name_suggests_money(str(cname)):
                continue
            if not _is_likely_numeric_aggregate_type(str(col.get("type", ""))):
                continue
            out.append((table_name, str(cname)))
    return out


def format_monetary_columns_for_llm(schema_details: Dict[str, List[dict]]) -> str:
    """One block listing monetary columns for SUM / AVG hints."""
    pairs = list_monetary_columns(schema_details)
    if not pairs:
        return (
            "MONETARY columns (for SUM/AVG when user asks amount, sales, revenue): "
            "none detected by name+numeric type — use numeric columns from the tables above."
        )
    lines = [
        "MONETARY columns (when the user asks amount, sales, revenue, or purchase totals, "
        "use SUM or AVG on one of these — NOT COUNT of rows unless they ask how many):",
    ]
    for table_name, col_name in pairs:
        lines.append(f"- `{table_name}`.`{col_name}`")
    return "\n".join(lines)


def _short_type(type_str: str) -> str:
    """Shorten long SQLAlchemy type strings for display."""
    s = str(type_str).strip()
    if len(s) > 72:
        return s[:69] + "..."
    return s


def get_table_names(engine: Engine) -> List[str]:
    """Return list of table names in the connected database."""
    try:
        inspector = inspect(engine)
        return inspector.get_table_names()
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Could not fetch table names: {exc}") from exc


def get_schema_details(engine: Engine) -> Dict[str, List[dict]]:
    """
    Build a nested structure of tables and columns (all columns from the DB — nothing omitted).

    Shape:
        { "customers": [ {"name", "type", "nullable"}, ... ], ... }
    """
    try:
        inspector = inspect(engine)
        schema_details: Dict[str, List[dict]] = {}

        for table_name in inspector.get_table_names():
            columns_info: List[dict] = []
            for col in inspector.get_columns(table_name):
                columns_info.append(
                    {
                        "name": col.get("name"),
                        "type": str(col.get("type")),
                        "nullable": bool(col.get("nullable", True)),
                    }
                )
            schema_details[table_name] = columns_info

        return schema_details
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Could not inspect schema: {exc}") from exc


def format_schema_structured_string(schema_details: Dict[str, List[dict]]) -> str:
    """
    Human / LLM-friendly schema (fixed layout).

    Table: customers
    Columns:
    - customer_id (INT)
    - name (VARCHAR)
    ...
    """
    if not schema_details:
        return "(No tables in schema.)"

    lines: List[str] = []
    for table_name in sorted(schema_details.keys()):
        lines.append(f"Table: {table_name}")
        lines.append("Columns:")
        for col in schema_details[table_name]:
            lines.append(f"- {col['name']} ({_short_type(col['type'])})")
        lines.append("")
    return "\n".join(lines).strip()


def format_fuzzy_hints_for_llm() -> str:
    """One block of text so the model can map words like 'sales' to real column names."""
    parts = ["Common word → prefer these column names IF they appear in the schema above:"]
    for word, cols in sorted(FUZZY_COLUMN_HINTS.items()):
        parts.append(f"- '{word}' → {', '.join(cols)}")
    return "\n".join(parts)


def schema_to_text(schema_details: Dict[str, List[dict]]) -> str:
    """Legacy compact format (still used if you need a simple dump)."""
    return format_schema_structured_string(schema_details)


def schema_text_or_placeholder(schema_details: Dict[str, List[dict]]) -> str:
    """Placeholder when the database has no tables."""
    if not schema_details:
        return "(No tables found in this database. Generate generic SELECT syntax only.)"
    return format_schema_structured_string(schema_details)


def build_schema_prompt_for_llm(engine: Engine) -> str:
    """
    Full schema string for Groq: structured tables + numeric hints + FK join hints.

    Inspector lists every column so the model should not miss names.
    """
    schema_details = get_schema_details(engine)
    if not schema_details:
        return "(No tables found in this database.)"

    try:
        inspector = inspect(engine)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Could not open schema inspector: {exc}") from exc

    lines: List[str] = [
        "You MUST use ONLY these tables and columns. Do not assume any column that is not listed.",
        "",
        format_schema_structured_string(schema_details),
        "",
        format_monetary_columns_for_llm(schema_details),
        "",
        "--- Extra hints (still only use names that appear above) ---",
    ]

    for table_name in sorted(schema_details.keys()):
        columns = schema_details[table_name]
        numeric_for_agg: List[str] = []
        for col in columns:
            if _is_likely_numeric_aggregate_type(str(col.get("type"))):
                numeric_for_agg.append(col["name"])

        if numeric_for_agg:
            lines.append(
                f"Table `{table_name}` — numeric columns suitable for SUM/AVG/COUNT(amount-like): "
                f"{', '.join(numeric_for_agg)}"
            )

        try:
            for fk in inspector.get_foreign_keys(table_name):
                cons = fk.get("constrained_columns") or []
                ref_t = fk.get("referred_table")
                ref_c = fk.get("referred_columns") or []
                if cons and ref_t and ref_c and len(cons) == len(ref_c):
                    pairs = " AND ".join(
                        f"{table_name}.{a} = {ref_t}.{b}"
                        for a, b in zip(cons, ref_c)
                    )
                    lines.append(f"Foreign key: {pairs}")
        except SQLAlchemyError:
            pass

    lines.append("")
    lines.append("METRIC mapping (match the user's words):")
    lines.append(
        "- amount / sales / revenue / purchase total / highest amount / total amount → "
        "SUM(monetary numeric column), alias AS total_amount or total_sales."
    )
    lines.append(
        "- how many / number of / count / total orders / order count → "
        "COUNT(order id column), alias AS total_orders or order_count."
    )
    lines.append(
        "- average / avg → AVG(monetary or numeric column), alias AS average_amount or avg_sales."
    )
    lines.append(
        "Never use COUNT(*) or COUNT(order_id) when the user asked for amount, sales, or revenue."
    )
    lines.append("")
    lines.append(format_fuzzy_hints_for_llm())
    return "\n".join(lines).strip()


def resolve_table_name(schema_details: Dict[str, List[dict]], name: str) -> Optional[str]:
    """Match table name case-insensitively; return canonical key or None."""
    if name in schema_details:
        return name
    lower = {k.lower(): k for k in schema_details}
    return lower.get(name.lower())
