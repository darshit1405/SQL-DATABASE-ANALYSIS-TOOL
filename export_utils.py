"""
Export helpers for query results (CSV and Excel).
"""

import io
from datetime import datetime

import pandas as pd
import streamlit as st


def _export_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def render_export_buttons(df: pd.DataFrame) -> None:
    """
    Show download buttons for CSV and Excel with timestamped filenames.

    Uses the current dataframe (original column names) for file bytes.
    If ``st.session_state["last_export_ts"]`` is set (after a successful run),
    filenames stay stable across Streamlit reruns.
    """
    if df.empty:
        st.info(
            "There are **no rows** to export. Run a query that returns data if you need a file."
        )
        return

    st.subheader("Export")

    ts = st.session_state.get("last_export_ts") or _export_timestamp()
    csv_name = f"query_result_{ts}.csv"
    xlsx_name = f"query_result_{ts}.xlsx"

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=csv_name,
        mime="text/csv",
        key="download_csv_btn",
    )

    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
    except Exception as exc:
        st.error(
            f"**Excel export failed.** Install dependencies (`pip install openpyxl`) or check "
            f"the error: {exc}"
        )
        return

    buffer.seek(0)
    st.download_button(
        label="Download Excel",
        data=buffer.getvalue(),
        file_name=xlsx_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_btn",
    )

    st.caption(
        "Downloads use timestamped names, e.g. `query_result_YYYYMMDD_HHMMSS.csv`. "
        "Click a button when you are ready — the file is prepared instantly."
    )
