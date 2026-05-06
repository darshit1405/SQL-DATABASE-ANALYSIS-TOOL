"""
Convert natural language to SQL using Groq API via LangChain.

Loads secrets from .env with python-dotenv (same folder as this file).
Expects GROQ_API_KEY in your environment or .env file.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional, Set

from dotenv import dotenv_values, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from query_executor import validate_read_only_sql
from intent_parser import parse_intent
import pandas as pd

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# Defaults use models that are still on Groq’s active list (see deprecations page).
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
FALLBACK_GROQ_MODELS: List[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

# --- Prompts: intent (COUNT vs SUM), meaningful SELECT aliases ---
SYSTEM_PROMPT = """You are an expert SQL generator. You MUST use ONLY the given schema.
Do NOT invent columns that are not listed. Return ONLY the SQL query — no markdown, no code fences.

INTENT — READ THE USER'S WORDS CAREFULLY:
- If they ask for amount, sales, revenue, purchase (money), highest amount, total amount, or money totals:
  use SUM(...) on the correct monetary numeric column from the schema.
  Never use COUNT when they want money totals.
- Use COUNT(...) only when they ask how many, number of, count, total orders (as a count), order count.
- Use AVG(...) when they ask average or avg (on the appropriate numeric column).

STRICT RULE (do not break this):
Never use COUNT when the user asks for amount, sales, revenue, or highest amount. Use SUM of the correct numeric monetary column from the SCHEMA.

METRIC → SQL pattern:
- sales / revenue / amount / total amount / highest amount → SUM(monetary_column) AS total_sales or total_amount
- orders / how many / number of / count / total orders → COUNT(order_id) AS total_orders or order_count
- average / avg → AVG(monetary_column) AS average_amount or avg_sales

ALIAS RULES:
- SUM(...) → AS total_amount, total_sales, or total_revenue (match the question).
- COUNT(...) → AS total_orders, order_count, or total_customers (match what is counted).
- AVG(...) → AS average_amount, avg_sales, avg_order_value.

OTHER RULES:
- Use ONLY table and column names from the schema.
- Use JOIN only when needed; follow foreign key hints in the schema when present.
- GROUP BY every non-aggregated column that appears in SELECT.
- One SELECT (or WITH ... SELECT) only. Never INSERT, UPDATE, DELETE, DROP, or DDL.

Always generate meaningful column aliases based on the calculation.
"""

FIX_SYSTEM_PROMPT = """You are an expert SQL debugger. Fix the SQL so it runs on the database described in the schema.
Return ONLY the corrected SQL query. No markdown. No explanation.
Use ONLY tables and columns from the schema. Preserve the user's intent.

If the user asked for amount, sales, or revenue, prefer SUM(monetary column), not COUNT.
If they asked how many or number of orders, prefer COUNT(...), not SUM.

Also use clear AS aliases for aggregates (total_orders, total_sales, average_amount, etc.) — not vague names like amount or value when the expression is COUNT/SUM/AVG.
"""

# Aliases we auto-replace when used after COUNT/SUM/AVG (too vague).
_GENERIC_AGGREGATE_ALIASES: Set[str] = frozenset(
    {
        "amount",
        "value",
        "data",
        "val",
        "values",
        "cnt",
        "num",
        "sum",
        "avg",
        "total",  # alone is vague for an aggregate result
        "result",
        "res",
        "col",
        "x",
    }
)

# Question wording → intent signals (lowercase regex on full question).
_RE_MONETARY_INTENT = re.compile(
    r"\b("
    r"amount|amounts|sales|revenue|gross|"
    r"purchase|purchases|proceeds|spent|"
    r"total\s+amount|highest\s+amount|total\s+sales|total\s+revenue|"
    r"money|payment|payments|paid"
    r")\b",
    re.IGNORECASE,
)

_RE_COUNT_INTENT = re.compile(
    r"\b("
    r"how\s+many|number\s+of|no\.\s*of|#?\s*of|"
    r"\bcount\b|order\s+count|total\s+orders|"
    r"orders?\s+per|customers?\s+per|per\s+customer|"
    r"quantity\s+of\s+orders"
    r")\b",
    re.IGNORECASE,
)

_RE_AVG_INTENT = re.compile(
    r"\b(average|avg\.?|mean)\b",
    re.IGNORECASE,
)

_RE_TOP_INTENT = re.compile(r"\b(top|highest|maximum|most)\b", re.IGNORECASE)
_RE_LOWEST_INTENT = re.compile(r"\b(lowest|minimum|least|bottom)\b", re.IGNORECASE)
_RE_TEMPORAL_INTENT = re.compile(r"\b(monthly|daily|yearly|weekly|per\s+month|per\s+day|per\s+year)\b", re.IGNORECASE)

_RE_SQL_HAS_COUNT = re.compile(r"\bCOUNT\s*\(", re.IGNORECASE)
_RE_SQL_HAS_SUM = re.compile(r"\bSUM\s*\(", re.IGNORECASE)
_RE_SQL_HAS_AVG = re.compile(r"\bAVG\s*\(", re.IGNORECASE)


def _infer_count_alias(inner: str) -> str:
    """Pick a readable alias for COUNT(...) based on the expression inside."""
    s = inner.lower().replace("`", "")
    if "*" in inner.strip():
        return "total_rows"
    if "distinct" in s and "order" in s:
        return "unique_orders"
    if "order" in s:
        return "total_orders"
    if "item" in s:
        return "total_items"
    if "customer" in s:
        return "total_customers"
    return "total_count"


def _infer_sum_alias(inner: str) -> str:
    """Pick a readable alias for SUM(...) based on inner columns."""
    s = inner.lower()
    if "sale" in s or "revenue" in s:
        return "total_sales"
    if "amount" in s or "price" in s or "cost" in s or "total_amount" in s:
        return "total_amount"
    return "total_sum"


def _infer_avg_alias(inner: str) -> str:
    s = inner.lower()
    if "sale" in s or "revenue" in s:
        return "avg_sales"
    if "amount" in s or "price" in s:
        return "average_amount"
    return "average_value"


def _infer_minmax_alias(func: str, inner: str) -> str:
    s = inner.lower()
    base = "min" if func.upper() == "MIN" else "max"
    if "date" in s or "time" in s:
        return f"{base}_date"
    if "amount" in s or "price" in s:
        return f"{base}_amount"
    return f"{base}_value"


def _improve_aggregate_aliases(sql: str) -> str:
    """
    After the LLM returns SQL, replace vague AS names on aggregates when needed.

    Example: COUNT(o.id) AS amount → COUNT(o.id) AS total_orders (when `amount` is generic).
    Uses simple rules so beginners can follow the logic.
    """
    # Match AGG( ... ) AS alias — innermost parens only (good enough for typical queries).
    pattern = re.compile(
        r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]*)\)\s+AS\s+(\w+)\b",
        re.IGNORECASE,
    )

    def replacer(match: re.Match) -> str:
        func = match.group(1).upper()
        inner = match.group(2).strip()
        alias = match.group(3)
        if alias.lower() not in _GENERIC_AGGREGATE_ALIASES:
            return match.group(0)

        if func == "COUNT":
            new_alias = _infer_count_alias(inner)
        elif func == "SUM":
            new_alias = _infer_sum_alias(inner)
        elif func == "AVG":
            new_alias = _infer_avg_alias(inner)
        elif func in ("MIN", "MAX"):
            new_alias = _infer_minmax_alias(func, inner)
        else:
            return match.group(0)

        # Keep original function casing from the query (COUNT vs count).
        return f"{match.group(1)}({inner}) AS {new_alias}"

    return pattern.sub(replacer, sql)


def _dedupe_select_aliases(sql: str) -> str:
    """
    If the same AS name appears twice in the SELECT list, suffix _2, _3.
    Very small safeguard against confusing duplicate aliases.
    """
    # Only touch "... AS name" in the first SELECT list (best-effort).
    m = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return sql
    select_body = m.group(1)
    # Nested SELECT in the column list — skip to avoid breaking subqueries.
    if re.search(r"\(\s*SELECT\b", select_body, re.IGNORECASE):
        return sql
    seen: Set[str] = set()
    counts: dict = {}

    def fix_alias(ma: re.Match) -> str:
        name = ma.group(1)
        key = name.lower()
        if key not in seen:
            seen.add(key)
            return ma.group(0)
        counts[key] = counts.get(key, 1) + 1
        suffix = counts[key]
        return f"AS {name}_{suffix}"

    new_body = re.sub(r"\bAS\s+(\w+)\b", fix_alias, select_body, flags=re.IGNORECASE)
    return sql[: m.start(1)] + new_body + sql[m.end(1) :]


def _get_groq_api_key() -> str:
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    file_vals = dotenv_values(_ENV_PATH)
    return (file_vals.get("GROQ_API_KEY") or "").strip()


def _get_groq_model_name() -> str:
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    model = os.getenv("GROQ_MODEL", "").strip()
    if model:
        return model
    file_vals = dotenv_values(_ENV_PATH)
    return (file_vals.get("GROQ_MODEL") or "").strip() or DEFAULT_GROQ_MODEL


def _extract_sql(raw_text: str) -> str:
    """Strip markdown, labels, and text before the first SELECT/WITH."""
    if not raw_text or not str(raw_text).strip():
        return ""

    text = str(raw_text).strip()

    if "```" in text:
        fence = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fence:
            text = fence.group(1).strip()
            kw = re.search(r"\b(WITH|SELECT)\b", text, flags=re.IGNORECASE | re.DOTALL)
            if kw:
                text = text[kw.start() :].strip()
            return text.rstrip().rstrip(";").strip()

    text = re.sub(r"^(?:SQL|Query)\s*:\s*", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

    kw = re.search(r"\b(WITH|SELECT)\b", text, flags=re.IGNORECASE | re.DOTALL)
    if kw:
        text = text[kw.start() :].strip()

    text = text.rstrip().rstrip(";").strip()

    lines = text.splitlines()
    kept: List[str] = []
    for line in lines:
        stripped = line.strip()
        if kept and re.match(
            r"^(Note|Here|This|The |I |We |Explanation|However)\b",
            stripped,
            re.IGNORECASE,
        ):
            break
        kept.append(line)
    text = "\n".join(kept).strip()
    return text.rstrip().rstrip(";").strip()


def _invoke_groq(system_text: str, user_text: str) -> str:
    """Call Groq and return raw model text (one successful completion)."""
    api_key = _get_groq_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your `.env` next to `app.py` and try again."
        )

    preferred = _get_groq_model_name()
    model_order = [preferred] + [m for m in FALLBACK_GROQ_MODELS if m != preferred]
    last_error: Optional[Exception] = None

    for model_name in model_order:
        try:
            llm = ChatGroq(
                model=model_name,
                groq_api_key=api_key,
                temperature=0,
                max_tokens=2048,
            )
            response = llm.invoke(
                [
                    SystemMessage(content=system_text),
                    HumanMessage(content=user_text),
                ]
            )
            return str(getattr(response, "content", None) or "")
        except Exception as exc:
            last_error = exc
            err_str = str(exc).lower()
            if (
                "401" in err_str
                or "403" in err_str
                or "unauthorized" in err_str
                or "invalid api key" in err_str
            ):
                raise RuntimeError(
                    "Groq rejected your API key. Check `GROQ_API_KEY` in `.env`."
                ) from exc
            if any(
                x in err_str
                for x in (
                    "404",
                    "model_not_found",
                    "unknown model",
                    "invalid model",
                    "not supported",
                    "decommissioned",
                    "model_decommissioned",
                    "no longer supported",
                )
            ):
                continue
            raise RuntimeError(
                "Could not reach the AI to fix SQL. Check your connection and try again. "
                f"Details: {exc}"
            ) from exc

    raise RuntimeError(
        "No working Groq model. Set `GROQ_MODEL` in `.env`. "
        + (f"Last error: {last_error}" if last_error else "")
    )


def fix_sql_using_schema_and_error(
    broken_sql: str,
    schema_text: str,
    question: str,
    error_message: str,
) -> str:
    """
    Ask Groq to repair SQL using schema + validation or database error text.

    Returns cleaned SQL that must still pass read-only checks.
    """
    user_prompt = f"""SCHEMA (only use these tables/columns):
---
{schema_text}
---

Original question (keep the same intent):
{question or "(not provided)"}

SQL that needs fixing:
{broken_sql}

Problem / error:
{error_message}

Output: one corrected SELECT query only. Use meaningful AS aliases for aggregates."""

    raw = _invoke_groq(FIX_SYSTEM_PROMPT, user_prompt)
    sql = _extract_sql(raw)
    if not sql:
        raise RuntimeError("The AI did not return usable SQL while trying to fix the query.")

    sql = _improve_aggregate_aliases(sql)
    sql = _dedupe_select_aliases(sql)

    ok, reason = validate_read_only_sql(sql)
    if not ok:
        raise RuntimeError(
            f"The AI returned SQL that is not allowed (read-only SELECT only): {reason}"
        )
    return sql


def _validate_generated_sql(sql: str) -> None:
    ok, reason = validate_read_only_sql(sql)
    if not ok:
        raise ValueError(
            "The model produced SQL that is not allowed here (only read-only SELECT). "
            f"{reason}"
        )


def _question_intent_flags(question: str) -> tuple:
    """Return (wants_monetary_aggregate, wants_count_aggregate, wants_avg_aggregate)."""
    q = (question or "").strip()
    if not q:
        return False, False, False
    monetary = bool(_RE_MONETARY_INTENT.search(q))
    count = bool(_RE_COUNT_INTENT.search(q))
    avg = bool(_RE_AVG_INTENT.search(q))
    return monetary, count, avg


def intent_mismatch_feedback(question: str, sql: str) -> Optional[str]:
    """
    If the SQL does not match obvious intent from the question, return instructions for the LLM.

    Returns None if OK (or ambiguous / skipped).
    """
    q = (question or "").strip()
    if not q:
        return None

    monetary = bool(_RE_MONETARY_INTENT.search(q))
    count = bool(_RE_COUNT_INTENT.search(q))
    avg = bool(_RE_AVG_INTENT.search(q))
    top = bool(_RE_TOP_INTENT.search(q))
    lowest = bool(_RE_LOWEST_INTENT.search(q))
    temporal = bool(_RE_TEMPORAL_INTENT.search(q))

    sql_u = (sql or "").upper()

    has_count = bool(_RE_SQL_HAS_COUNT.search(sql_u))
    has_sum = bool(_RE_SQL_HAS_SUM.search(sql_u))
    has_avg = bool(_RE_SQL_HAS_AVG.search(sql_u))

    issues: List[str] = []

    # If both monetary and count are asked, do not enforce one-sided rules.
    if monetary and not count:
        if has_count and not has_sum and not has_avg:
            issues.append(
                "CORRECTION REQUIRED: The user asked for amount, sales, revenue, or similar. "
                "Your SQL used COUNT but should use SUM on the correct monetary column from the SCHEMA "
                "(see MONETARY columns). Do not use COUNT for money totals. "
                "Alias the SUM as total_amount or total_sales."
            )

    if count and not monetary:
        if has_sum and not has_count:
            issues.append(
                "CORRECTION REQUIRED: The user asked how many, number of, count, or total orders. "
                "Your SQL used SUM but should use COUNT (e.g. COUNT(order_id)) for counting rows. "
                "Alias as total_orders or order_count."
            )

    if avg and not monetary and not count:
        if not has_avg:
            issues.append(
                "CORRECTION REQUIRED: The user asked for an average. "
                "Use AVG(...) on the appropriate numeric column from the SCHEMA. "
                "Alias as average_amount or avg_sales."
            )

    if top:
        if "ORDER BY" not in sql_u or "DESC" not in sql_u or "LIMIT" not in sql_u:
            issues.append(
                "CORRECTION REQUIRED: The user asked for top/highest/maximum. "
                "Your SQL must use ORDER BY ... DESC and LIMIT to get the top results."
            )

    if lowest:
        if "ORDER BY" not in sql_u or "LIMIT" not in sql_u:
            issues.append(
                "CORRECTION REQUIRED: The user asked for lowest/minimum/bottom. "
                "Your SQL must use ORDER BY ... ASC and LIMIT to get the lowest results."
            )
        elif "DESC" in sql_u and "ASC" not in sql_u:
            issues.append(
                "CORRECTION REQUIRED: The user asked for lowest/minimum/bottom. "
                "Your SQL used DESC instead of ASC. Use ORDER BY ... ASC and LIMIT."
            )

    if temporal:
        if "GROUP BY" not in sql_u:
            issues.append(
                "CORRECTION REQUIRED: The user asked for a monthly/daily/yearly trend. "
                "Your SQL must extract the date part and use GROUP BY to show the trend."
            )

    if issues:
        return "\n".join(issues)

    return None


def generate_sql_from_question(question: str, schema_text: str) -> str:
    """
    Send schema + question to Groq; return one validated SELECT-style SQL string.

    Retries up to 2 times if the SQL mismatches clear COUNT vs SUM vs AVG intent.
    """
    if not _get_groq_api_key():
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your `.env` file next to `app.py`, then try again."
        )

    if not question or not str(question).strip():
        raise ValueError("Please type a question before generating SQL.")

    schema_block = (schema_text or "").strip()
    if not schema_block:
        schema_block = (
            "(No schema loaded — connect to the database and detect schema first.)"
        )

    few_shot = """
Examples (your real SCHEMA section replaces table/column names as needed):

Question: Give me top 2 cities with highest amount
SQL:
SELECT c.city, SUM(o.amount) AS total_amount
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.city
ORDER BY total_amount DESC
LIMIT 2;

Question: Give me top 2 cities by number of orders
SQL:
SELECT c.city, COUNT(o.order_id) AS total_orders
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.city
ORDER BY total_orders DESC
LIMIT 2;

More examples:

Question: Show all customers
SELECT * FROM customers;

Question: Show total sales and number of orders by city
SELECT c.city,
       COUNT(o.order_id) AS total_orders,
       SUM(o.total_amount) AS total_sales
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.city;

Question: Total sales by city
SELECT c.city, SUM(o.total_amount) AS total_sales
FROM customers AS c
INNER JOIN orders AS o ON c.customer_id = o.customer_id
GROUP BY c.city;

Question: How many orders per customer
SELECT c.name, COUNT(o.order_id) AS total_orders
FROM customers AS c
LEFT JOIN orders AS o ON c.customer_id = o.customer_id
GROUP BY c.name;
"""

    max_retries = 2
    correction_note = ""

    # Parse intent to pass structured data to LLM
    parsed_intent_data = parse_intent(question.strip())
    intent_block = ""
    if parsed_intent_data:
        intent_block = f"""
PARSED INTENT:
The user input was parsed into the following structured intent.
Use this JSON to guide your SQL generation (metric, grouping, limits, order):
{json.dumps(parsed_intent_data, indent=2)}

"""

    for attempt in range(max_retries + 1):
        correction_block = ""
        if correction_note:
            correction_block = f"""
IMPORTANT — FIX YOUR PREVIOUS ANSWER:
{correction_note}

"""

        user_prompt = f"""SCHEMA:
---
{schema_block}
---

{few_shot}

{intent_block}{correction_block}USER INPUT:
{question.strip()}

Answer with one SQL query only (no markdown). Use ONLY names from SCHEMA above.
Use meaningful AS aliases for every aggregate (COUNT, SUM, AVG)."""

        raw = _invoke_groq(SYSTEM_PROMPT, user_prompt)
        sql = _extract_sql(raw)
        if not sql:
            raise RuntimeError(
                "The AI returned no usable SQL. Try naming a table or metric more clearly."
            )

        sql = _improve_aggregate_aliases(sql)
        sql = _dedupe_select_aliases(sql)

        _validate_generated_sql(sql)

        mismatch = intent_mismatch_feedback(question, sql)
        if mismatch is None:
            return sql

        if attempt >= max_retries:
            raise RuntimeError(
                "Could not generate SQL that matches your question intent after retries. "
                f"Last issue: {mismatch}"
            )

        correction_note = mismatch

    raise RuntimeError("SQL generation failed after intent retries.")

def normalize_user_query(question: str) -> str:
    """
    Translate and normalize Hinglish/Hindi/Gujarati/broken English into a clean English business intent.
    If already English, standardizes the business terms.
    """
    if not _get_groq_api_key():
        return question # Fallback if no key

    system_text = """You are a multilingual language understanding layer.
The user might ask for data in English, Hinglish, Hindi, or Gujarati.
Your job is to translate and normalize their request into a clear, concise English business intent for a SQL generator.

Rules for synonyms:
- sales, kharidi, kharch, vikri, વેંચાણ, રકમ -> amount or revenue
- sabse jyada, સૌથી વધારે -> top or highest
- sabse kam, સૌથી ઓછું -> lowest or bottom
- mahina, મહિનો -> month
- sheher, શહેર -> city
- grahak, ગ્રાહક -> customer
- aur, ane, અને -> and

If the query is already clear English, just return it slightly improved or as-is.
If the query is very unclear or meaningless, output: "UNCLEAR_QUERY"

Examples:
- "sabse jyada sales kis city ma che" -> "top cities by sales amount"
- "pichla 30 din ka revenue" -> "revenue for the last 30 days"
- "top 5 city sales" -> "top 5 cities by sales"

Output ONLY the translated, normalized sentence. No explanations. No quotes."""
    
    try:
        norm = _invoke_groq(system_text, f"USER QUERY: {question}").strip()
        if "UNCLEAR_QUERY" in norm.upper() or not norm:
            return question
        return norm
    except Exception:
        return question

def generate_insight_summary(question: str, df: pd.DataFrame) -> str:
    """
    Generate a 1-3 line non-technical summary of the data in df based on the user's question.
    """
    if df is None or df.empty:
        return "No data returned."
        
    if not _get_groq_api_key():
        return "Data retrieved successfully."

    # Send only first 10 rows to save context and avoid huge prompts
    sample_df = df.head(10).to_csv(index=False)
    
    system_text = """You are an expert data analyst. 
The user asked a question, and the database returned the following data (first 10 rows).
Your job is to write a 1-3 line simple, non-technical Insight Summary of the results.

RULES:
- Explain the result in 1 to 3 short sentences.
- Be friendly, professional, and clear for a non-technical business user.
- Mention the highest/best or lowest/worst value if it is relevant.
- Do not mention SQL, databases, or 'Dataframe'.
- Output ONLY the summary text."""

    user_text = f"User Question: {question}\n\nData Result:\n{sample_df}"
    
    try:
        summary = _invoke_groq(system_text, user_text).strip()
        return summary
    except Exception:
        return "Data retrieved successfully."
