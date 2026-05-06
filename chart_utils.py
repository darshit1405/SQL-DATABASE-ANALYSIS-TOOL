"""
Plotly chart helpers for query results in Streamlit.

Chart type drives which controls appear (bar/line: X+Y; pie: label + value).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from decimal import Decimal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def prepare_chart_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up string formatting and converts apparent numeric columns to actual numerics.
    """
    new_df = df.copy()
    target_names = ["total_sales", "total_amount", "revenue", "amount", "price", "avg_order_value", "total_orders", "count"]
    
    for col in new_df.columns:
        s = new_df[col]
        
        def clean_val(x):
            if isinstance(x, str):
                return x.replace(",", "").strip()
            if isinstance(x, Decimal):
                return float(x)
            return x
            
        if s.dtype == object:
            s_clean = s.apply(clean_val)
        else:
            s_clean = s
            
        s_num = pd.to_numeric(s_clean, errors="coerce")
        
        non_null_count = s_clean.notna().sum()
        is_target = any(t in str(col).lower() for t in target_names)
        
        if non_null_count > 0:
            success_count = s_num.notna().sum()
            # If 60% convert successfully OR it's a known numeric target name
            if success_count / non_null_count >= 0.6 or is_target:
                new_df[col] = s_num
        elif is_target:
            new_df[col] = s_num
            
    return new_df


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns numeric, categorical, and datetime column lists.
    First cleans the dataframe to ensure numbers are detected.
    """
    df_prepared = prepare_chart_dataframe(df)
    
    numeric: List[str] = []
    categorical: List[str] = []
    datetime_cols: List[str] = []

    for col in df_prepared.columns:
        s = df_prepared[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(col)
            continue
        
        if s.dtype == object:
            sample = s.dropna().head(50)
            if not sample.empty:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.7:
                    datetime_cols.append(col)
                    continue
        categorical.append(col)

    return numeric, categorical, datetime_cols


def _pick_default_index(options: List[str], preferred: List[str]) -> int:
    for p in preferred:
        if p in options:
            return options.index(p)
    return 0


def _sync_chart_config(
    chart_type: str,
    x_or_label: Optional[str] = None,
    y_or_value: Optional[str] = None,
) -> None:
    """Store last chart picks for query history (session only)."""
    payload: Dict[str, Optional[str]] = {"chart_type": chart_type}
    if chart_type == "Pie chart":
        payload["label_col"] = x_or_label
        payload["value_col"] = y_or_value
    else:
        payload["x_col"] = x_or_label
        payload["y_col"] = y_or_value
    st.session_state["last_chart_config"] = payload


def render_chart_section(df: pd.DataFrame, numeric_cols: List[str] = None, cat_cols: List[str] = None, dt_cols: List[str] = None) -> None:
    """
    Chart UI and Plotly figure. Updates ``st.session_state["last_chart_config"]``.
    Uses st.status while building the figure so the UI feels responsive.
    """
    if df.empty:
        st.info(
            "Your query returned **no rows**, so there is nothing to chart. "
            "Try different filters or check your data."
        )
        _sync_chart_config("Bar chart", None, None)
        return

    all_cols = df.columns.tolist()
    if len(all_cols) < 2:
        st.info(
            "You need **at least two columns** (for example category + number) to build a chart."
        )
        _sync_chart_config("Bar chart", None, None)
        return

    st.subheader("Visualization")

    if numeric_cols is None or cat_cols is None or dt_cols is None:
        numeric_cols, cat_cols, dt_cols = get_column_types(df)

    # Restore picks from "View result" in history (one-shot).
    restore = st.session_state.pop("_chart_restore", None) or {}
    restored_type = restore.get("chart_type")

    chart_options = ["Bar chart", "Line chart", "Pie chart"]
    type_index = 0
    if restored_type in chart_options:
        type_index = chart_options.index(restored_type)

    chart_type = st.selectbox(
        "Chart type",
        chart_options,
        index=type_index,
        key="chart_type_select",
    )

    # --- Pie: label + value (no X/Y axis titles on figure) ---
    if chart_type == "Pie chart":
        st.caption(
            "Pie chart requires **one label column** and **one numeric value column**."
        )

        preferred_lbl = dt_cols + cat_cols + [c for c in all_cols if c not in numeric_cols]
        label_default = _pick_default_index(all_cols, preferred_lbl)
        if restore.get("label_col") in all_cols:
            label_default = all_cols.index(restore["label_col"])

        value_candidates = numeric_cols if numeric_cols else all_cols
        if not value_candidates:
            st.warning(
                "Pie chart needs **one text-like column and one numeric column**. "
                "No numeric column was detected — try a Bar chart or fix the query."
            )
            _sync_chart_config(chart_type, None, None)
            return

        value_default = _pick_default_index(all_cols, value_candidates)
        if restore.get("value_col") in all_cols:
            value_default = all_cols.index(restore["value_col"])

        label_col = st.selectbox(
            "Label column (slice names)",
            all_cols,
            index=label_default,
            key="pie_label_select",
        )
        value_choices = [c for c in value_candidates if c != label_col] or value_candidates
        value_index = _pick_default_index(
            value_choices,
            [restore["value_col"]] if restore.get("value_col") else numeric_cols,
        )
        value_col = st.selectbox(
            "Value column (must be numeric)",
            value_choices,
            index=min(value_index, len(value_choices) - 1),
            key="pie_value_select",
        )

        if label_col == value_col:
            st.warning("Choose **two different columns** for label and value.")
            _sync_chart_config(chart_type, label_col, value_col)
            return

        y_data = df[value_col]
        y_err = None if pd.api.types.is_numeric_dtype(y_data) else f"Column `{value_col}` has no usable numeric values."

        if not pd.api.types.is_numeric_dtype(y_data) and y_data.notna().any():
            # Already prepared df so it should be numeric. If it isn't, there is a hard failure.
            st.warning(
                "Pie chart needs a **numeric value column**. "
                f"{y_err or 'Pick a column with numbers.'}"
            )
            _sync_chart_config(chart_type, label_col, value_col)
            return

        with st.status("creating charts...", expanded=False) as status:
            status.write("creating charts...")
            status.write("Rendering chart...")
            pie_df = pd.DataFrame({label_col: df[label_col], value_col: y_data}).dropna(
                subset=[value_col]
            )
            if pie_df.empty:
                st.warning("No numeric values left after cleaning — adjust your columns.")
                status.update(label="Could not render", state="error")
                _sync_chart_config(chart_type, label_col, value_col)
                return

            if (pie_df[value_col] < 0).any():
                st.warning(
                    "Some values are negative. Pie charts work best with positive numbers."
                )

            try:
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=pie_df[label_col].astype(str),
                            values=pie_df[value_col],
                            hole=0.35,
                        )
                    ]
                )
                fig.update_layout(
                    margin=dict(l=24, r=24, t=32, b=24),
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
                status.update(label="Chart ready", state="complete")
            except Exception as exc:
                st.warning(f"Could not build the pie chart. {exc}")
                status.update(label="Chart failed", state="error")

        _sync_chart_config(chart_type, label_col, value_col)
        return

    # --- Bar / Line: X and Y ---
    preferred_x_bar = dt_cols + cat_cols + [c for c in all_cols if c not in numeric_cols]
    preferred_x_line = dt_cols + preferred_x_bar
    
    # Requirement: For line chart, if total_sales exists, it must appear in Y-axis dropdown
    # We will prioritize target columns for Y.
    target_names = ["total_sales", "total_amount", "revenue", "amount", "price", "avg_order_value", "total_orders", "count"]
    priority_y = [c for c in numeric_cols if any(t in str(c).lower() for t in target_names)]
    preferred_y = priority_y + [c for c in numeric_cols if c not in priority_y] if numeric_cols else all_cols

    if chart_type == "Line chart":
        preferred_x = preferred_x_line
    else:
        preferred_x = preferred_x_bar

    x_default = _pick_default_index(all_cols, preferred_x)
    if restore.get("x_col") in all_cols:
        x_default = all_cols.index(restore["x_col"])

    x_axis = st.selectbox(
        "X-axis column",
        all_cols,
        index=x_default,
        key="chart_x_select",
    )

    y_options = [c for c in all_cols if c != x_axis and c in numeric_cols]
    if not y_options:
        y_options = [c for c in all_cols if c != x_axis]
    if not y_options:
        y_options = all_cols

    y_default = _pick_default_index(y_options, preferred_y)
    if restore.get("y_col") in y_options:
        y_default = y_options.index(restore["y_col"])

    y_axis = st.selectbox(
        "Y-axis column (numeric)",
        y_options,
        index=min(y_default, len(y_options) - 1),
        key="chart_y_select",
    )

    if chart_type == "Line chart":
        if not numeric_cols:
            st.warning(
                "Line chart needs at least **one numeric column** for Y. "
                "None detected — try a different query or use a Bar chart."
            )
            _sync_chart_config(chart_type, x_axis, y_axis)
            return
        if x_axis not in dt_cols and x_axis not in numeric_cols:
            st.caption(
                "Tip: Line charts read best when X is **date/time** or an ordered field."
            )

    if x_axis == y_axis:
        st.warning("Choose **two different columns** for X and Y.")
        _sync_chart_config(chart_type, x_axis, y_axis)
        return

    y_data = df[y_axis]
    y_err = None if pd.api.types.is_numeric_dtype(y_data) else f"Column `{y_axis}` has no usable numeric values."
    
    if not pd.api.types.is_numeric_dtype(y_data) and y_data.notna().any():
        st.warning(
            f"**{chart_type}** needs a numeric Y column. {y_err or 'Pick a numeric column.'}"
        )
        _sync_chart_config(chart_type, x_axis, y_axis)
        return

    with st.status("creating charts...", expanded=False) as status:
        status.write("creating charts...")
        plot_df = pd.DataFrame({x_axis: df[x_axis], y_axis: y_data}).dropna(subset=[y_axis])
        if plot_df.empty:
            st.warning("After removing empty values, there is nothing left to plot.")
            status.update(label="No data to plot", state="error")
            _sync_chart_config(chart_type, x_axis, y_axis)
            return

        try:
            status.write("Rendering chart…")
            if chart_type == "Bar chart":
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=plot_df[x_axis].astype(str),
                            y=plot_df[y_axis],
                            name=y_axis,
                            marker_color="#0e7490",
                        )
                    ]
                )
                fig.update_layout(
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    margin=dict(l=48, r=32, t=40, b=80),
                    template="plotly_white",
                )
            else:
                sort_df = plot_df.copy()
                if x_axis in dt_cols or pd.api.types.is_datetime64_any_dtype(sort_df[x_axis]):
                    sort_df[x_axis] = pd.to_datetime(sort_df[x_axis], errors="coerce")
                    sort_df = sort_df.sort_values(x_axis)
                else:
                    try:
                        sort_df["_sk"] = pd.to_numeric(sort_df[x_axis], errors="coerce")
                        if sort_df["_sk"].notna().all():
                            sort_df = sort_df.sort_values("_sk")
                    except Exception:
                        pass

                fig = go.Figure(
                    data=[
                        go.Scatter(
                            x=sort_df[x_axis].astype(str),
                            y=sort_df[y_axis],
                            mode="lines+markers",
                            name=y_axis,
                            line=dict(color="#0369a1", width=2),
                        )
                    ]
                )
                fig.update_layout(
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    margin=dict(l=48, r=32, t=40, b=80),
                    template="plotly_white",
                )

            st.plotly_chart(fig, use_container_width=True)
            status.update(label="Chart ready", state="complete")
        except Exception as exc:
            st.warning(f"Could not build the chart. {exc}")
            status.update(label="Chart failed", state="error")

    _sync_chart_config(chart_type, x_axis, y_axis)


def render_auto_dashboard(df: pd.DataFrame, num_cols: List[str] = None, cat_cols: List[str] = None, dt_cols: List[str] = None) -> None:
    """
    Automatically suggest and generate useful visualizations and metric cards 
    based on result columns.
    """
    if df.empty or len(df.columns) < 2:
        st.info("Not enough data to auto-generate a dashboard. Try adjusting your query.")
        return

    with st.status("Analyzing result data...", expanded=True) as status:
        if num_cols is None or cat_cols is None or dt_cols is None:
            num_cols, cat_cols, dt_cols = get_column_types(df)
            
        status.write("Generating dashboard...")

        # 1. Render Metric Cards for numeric totals
        if num_cols:
            st.markdown("### Totals")
            # Limit to 4 cards to keep it tidy
            cols = st.columns(min(len(num_cols), 4))
            for i, col in enumerate(num_cols[:4]):
                total = pd.to_numeric(df[col], errors="coerce").sum()
                if pd.isna(total):
                    total_str = "0"
                elif total == int(total):
                    total_str = f"{int(total):,}"
                else:
                    total_str = f"{total:,.2f}"
                cols[i].metric(label=f"Total {col}", value=total_str)

        # 2. Decide on best chart
        chart_type = None
        x_col = None
        y_col = None

        if not num_cols:
            status.update(label="Analysis complete", state="complete")
            st.info("No numeric columns found. Only metric cards can be generated for text data if modified.")
            return

        y_col = num_cols[0]

        # Check for temporal columns masquerading as strings (e.g. "month_name")
        time_col = None
        if dt_cols:
            time_col = dt_cols[0]
        else:
            for c in cat_cols:
                if any(k in c.lower() for k in ["date", "month", "year", "time", "day", "week"]):
                    time_col = c
                    break

        if time_col:
            chart_type = "Line chart"
            x_col = time_col
        elif cat_cols and len(df) <= 10:
            chart_type = "Pie chart"
            x_col = cat_cols[0]
        elif cat_cols:
            chart_type = "Bar chart"
            x_col = cat_cols[0]
        else:
            chart_type = "Bar chart"
            x_col = df.columns[0] if df.columns[0] != y_col else df.columns[1]

        # 3. Render the Auto Chart
        if chart_type and x_col and y_col and x_col != y_col:
            st.markdown(f"### Auto-Generated: {chart_type}")
            try:
                if chart_type == "Pie chart":
                    pie_df = df.dropna(subset=[y_col])
                    fig = go.Figure(
                        data=[
                            go.Pie(
                                labels=pie_df[x_col].astype(str),
                                values=pd.to_numeric(pie_df[y_col], errors='coerce'),
                                hole=0.35,
                            )
                        ]
                    )
                    fig.update_layout(margin=dict(l=24, r=24, t=32, b=24))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"This chart shows {y_col} by {x_col}.")

                elif chart_type == "Line chart":
                    sort_df = df.dropna(subset=[y_col]).copy()
                    if x_col in dt_cols or pd.api.types.is_datetime64_any_dtype(sort_df[x_col]):
                        sort_df[x_col] = pd.to_datetime(sort_df[x_col], errors="coerce")
                        sort_df = sort_df.sort_values(x_col)
                    fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=sort_df[x_col].astype(str),
                                y=pd.to_numeric(sort_df[y_col], errors='coerce'),
                                mode="lines+markers",
                                name=y_col,
                                line=dict(color="#0369a1", width=2),
                            )
                        ]
                    )
                    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, margin=dict(l=48, r=32, t=40, b=80), template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"This chart shows the trend of {y_col} over {x_col}.")

                elif chart_type == "Bar chart":
                    plot_df = df.dropna(subset=[y_col])
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=plot_df[x_col].astype(str),
                                y=pd.to_numeric(plot_df[y_col], errors='coerce'),
                                name=y_col,
                                marker_color="#0e7490",
                            )
                        ]
                    )
                    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, margin=dict(l=48, r=32, t=40, b=80), template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"This chart shows {y_col} by {x_col}.")
            except Exception as e:
                st.warning(f"Could not build auto chart. {e}")
                
        status.update(label="Dashboard ready", state="complete")
