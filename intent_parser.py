import re
from typing import Any, Dict

def parse_intent(user_input: str) -> Dict[str, Any]:
    """
    Parse raw user input into a structured intent dictionary.
    This acts as a smart keyword mapping layer to guide the LLM.
    
    If parsing fails to detect anything, it returns an empty dict,
    and the LLM will fallback to normal question-based generation.
    """
    text = user_input.lower().strip()
    intent: Dict[str, Any] = {}

    # 1. Detect Limit and Order (e.g. "top 5", "lowest 10")
    limit_desc_match = re.search(r'\b(?:top|highest|maximum|most|first)\s+(\d+)\b', text)
    if limit_desc_match:
        intent["limit"] = int(limit_desc_match.group(1))
        intent["order"] = "DESC"
    else:
        limit_asc_match = re.search(r'\b(?:lowest|minimum|least|bottom)\s+(\d+)\b', text)
        if limit_asc_match:
            intent["limit"] = int(limit_asc_match.group(1))
            intent["order"] = "ASC"
        # Fallback if no number is given but keyword exists
        elif re.search(r'\b(?:top|highest|maximum|most)\b', text):
            intent["order"] = "DESC"
        elif re.search(r'\b(?:lowest|minimum|least|bottom)\b', text):
            intent["order"] = "ASC"

    # 2. Detect Metric and Aggregation
    # sales -> amount (SUM), orders -> order_id (COUNT)
    if re.search(r'\b(?:sales|revenue|amount|spent|spending|purchase|proceeds)\b', text):
        intent["metric"] = "sales"
        intent["aggregation"] = "SUM"
    elif re.search(r'\b(?:orders?|count|how many|number of)\b', text):
        intent["metric"] = "orders"
        intent["aggregation"] = "COUNT"
    elif re.search(r'\b(?:customers?|users?|clients?)\b', text):
        # We store the metric but aggregation depends on context (often COUNT)
        intent["metric"] = "customers"

    # Check for average explicitly
    if re.search(r'\b(?:average|avg|mean)\b', text):
        intent["aggregation"] = "AVG"

    # 3. Detect Grouping (e.g., "by city", "per customer", "monthly")
    group_match = re.search(r'\b(?:by|per)\s+([a-z0-9_]+)\b', text)
    if group_match:
        intent["group_by"] = group_match.group(1)

    # Special temporal grouping
    if re.search(r'\b(?:monthly|per month|month)\b', text):
        intent["group_by"] = "month"
    elif re.search(r'\b(?:daily|per day|day)\b', text):
        intent["group_by"] = "day"
    elif re.search(r'\b(?:yearly|per year|year)\b', text):
        intent["group_by"] = "year"

    # 4. Detect Filters (e.g., "last 30 days")
    filter_match = re.search(r'(last\s+\d+\s+days?)', text)
    if filter_match:
        intent["filters"] = filter_match.group(1)

    return intent
