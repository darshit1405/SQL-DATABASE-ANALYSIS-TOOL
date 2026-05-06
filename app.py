"""
Streamlit app for SQL Database Analysis Tool.

Flow: connect (sidebar) → ask in English → Groq generates SQL → review & run SELECT
→ results, charts, export, and session query history.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from config import Settings
from database import create_db_engine, safe_dispose, test_connection
from llm_sql_generator import generate_sql_from_question
from query_executor import execute_sql_query_with_fixes, validate_read_only_sql
from schema_utils import build_schema_prompt_for_llm, get_schema_details, get_table_names
from chart_utils import render_chart_section, render_auto_dashboard, prepare_chart_dataframe, get_column_types
from export_utils import render_export_buttons

try:
    from streamlit_mic_recorder import speech_to_text
except ImportError:
    speech_to_text = None

MAX_HISTORY_ENTRIES = 50


def _inject_app_styles() -> None:
    """Dashboard look: cards, spacing, buttons, history hover."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f8fafc;
        }
        .app-header-wrap {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 2rem 1.5rem;
            border-radius: 16px;
            margin: 0 0 1.5rem 0;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .app-header-wrap h1 {
            color: #ffffff !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .app-header-wrap p {
            color: #94a3b8 !important;
            margin: 0.5rem 0 0 0 !important;
            font-size: 1.1rem;
        }
        
        /* Progress Steps */
        .step-done { color: #10b981; font-weight: 600; text-align: center; }
        .step-active { color: #0ea5e9; font-weight: 700; text-align: center; }
        .step-pending { color: #94a3b8; text-align: center; }
        
        /* Sidebar History Cards */
        div[data-testid="stSidebarContent"] .history-card {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
            background: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: all 0.2s ease-in-out;
            cursor: default;
        }
        div[data-testid="stSidebarContent"] .history-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
            border-color: #cbd5e1;
        }
        
        /* Main body cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px !important;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            padding: 1rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 8px 20px rgba(0,0,0,0.06);
            transform: translateY(-1px);
        }
        
        /* Buttons */
        div[data-testid="stBaseButton-primary"] button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease;
        }
        div[data-testid="stBaseButton-primary"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
        }
        div[data-testid="stBaseButton-secondary"] button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease;
        }
        div[data-testid="stBaseButton-secondary"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    """Defaults for session_state keys (beginner-friendly single place)."""
    defaults: Dict[str, Any] = {
        "db_connected": False,
        "engine": None,
        "schema_text": "",
        "sql_editor": "",
        "last_error": "",
        "query_result_df": None,
        "schema_map": None,
        "query_history": [],
        "history_next_id": 1,
        "active_history_id": None,
        "last_chart_config": {},
        "last_run_sql": "",
        "last_export_ts": "",
        "user_query": "",
        "normalized_query": "",
        "insight_summary": "",
        "setting_advanced_mode": False,
        "setting_auto_chart": True,
        "setting_history": True,
        "setting_lang_assist": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _db_connected() -> bool:
    return st.session_state.get("db_connected", False)


def _settings_display() -> Settings:
    """Load `.env` settings for display."""
    return Settings()


def _humanize_column_name(name: str) -> str:
    """Turn snake_case into Title Case for display only."""
    parts = str(name).replace("__", "_").split("_")
    return " ".join(p.capitalize() for p in parts if p)


def _dataframe_display_copy(df: pd.DataFrame) -> pd.DataFrame:
    """Same data, prettier column labels for the UI table."""
    rename = {c: _humanize_column_name(c) for c in df.columns}
    return df.rename(columns=rename)


def _render_numeric_summary(df: pd.DataFrame) -> None:
    num = df.select_dtypes(include=["number"])
    if num.empty:
        return
    with st.expander("🔢 Numeric summary (statistics)", expanded=False):
        st.dataframe(
            num.describe().T,
            use_container_width=True,
            hide_index=False,
        )


def _sync_history_chart_from_session() -> None:
    """Attach latest chart picks to the newest history row if it matches the last run."""
    active_id = st.session_state.get("active_history_id")
    if not active_id:
        return
    for e in st.session_state.query_history:
        if e["id"] == active_id:
            e["chart"] = dict(st.session_state.get("last_chart_config") or {})
            break


def _append_query_history(
    question: str,
    final_sql: str,
    result_df: pd.DataFrame,
    insight: str = "",
) -> None:
    """Save a successful run into session history (max MAX_HISTORY_ENTRIES)."""
    entry = {
        "id": st.session_state.history_next_id,
        "question": question.strip() or "(no question text)",
        "sql": final_sql.strip(),
        "timestamp": datetime.now(),
        "row_count": len(result_df),
        "col_count": len(result_df.columns),
        "df": result_df.copy(),
        "chart": {},
        "insight": insight,
    }
    st.session_state.history_next_id += 1
    st.session_state.query_history.insert(0, entry)
    if len(st.session_state.query_history) > MAX_HISTORY_ENTRIES:
        st.session_state.query_history = st.session_state.query_history[:MAX_HISTORY_ENTRIES]
    st.session_state.active_history_id = entry["id"]
    st.session_state.last_run_sql = final_sql.strip()
    st.session_state.last_export_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- Callbacks for sidebar history (avoid rerun quirks) ---
def _reuse_question_cb(text: str) -> None:
    st.session_state["user_query"] = text


def _reuse_sql_cb(text: str) -> None:
    st.session_state.sql_editor = text


def _view_result_cb(entry_id: int) -> None:
    for e in st.session_state.query_history:
        if e["id"] == entry_id:
            st.session_state.query_result_df = e["df"].copy()
            st.session_state.sql_editor = e["sql"]
            st.session_state.active_history_id = entry_id
            st.session_state._chart_restore = dict(e.get("chart") or {})
            st.session_state.insight_summary = e.get("insight", "")
            ts = e.get("timestamp")
            if isinstance(ts, datetime):
                st.session_state.last_export_ts = ts.strftime("%Y-%m-%d %H:%M:%S")
            st.toast("Loaded saved result.")
            return


def _delete_entry_cb(entry_id: int) -> None:
    st.session_state.query_history = [
        e for e in st.session_state.query_history if e["id"] != entry_id
    ]


def _clear_history_cb() -> None:
    st.session_state.query_history = []


def _clear_result_cb() -> None:
    st.session_state.query_result_df = None
    st.session_state.active_history_id = None
    st.session_state.pop("_chart_restore", None)


def _reset_app_cb() -> None:
    safe_dispose(st.session_state.engine)
    st.session_state.engine = None
    st.session_state.db_connected = False
    st.session_state.schema_text = ""
    st.session_state.schema_map = None
    st.session_state.sql_editor = ""
    st.session_state["user_query"] = ""
    st.session_state.query_result_df = None
    st.session_state.last_error = ""
    st.session_state.query_history = []
    st.session_state.history_next_id = 1
    st.session_state.active_history_id = None
    st.session_state.last_chart_config = {}
    st.session_state.pop("_chart_restore", None)
    st.session_state.last_run_sql = ""
    st.session_state.last_export_ts = ""
    st.session_state.insight_summary = ""
    st.toast("Session reset.")


def _short_type_ui(type_str: str) -> str:
    t = str(type_str).strip()
    return t if len(t) <= 48 else t[:45] + "…"


def _render_sidebar() -> None:
    """Project sidebar: status, connection, schema, history, clears."""
    s = _settings_display()

    st.sidebar.markdown("### 🔌 App Status")
    if _db_connected():
        st.sidebar.success("Database Connected ✅")
        st.sidebar.caption(f"**Connected to:** `{s.db_name or '—'}`")
        table_n = len(st.session_state.schema_map) if st.session_state.schema_map else 0
        st.sidebar.caption(f"**Tables detected:** {table_n}")
    else:
        st.sidebar.error("Database Not Connected ❌")

    st.sidebar.divider()
    
    st.sidebar.markdown("### ⚙️ Database Connection")
    if st.sidebar.button("Connect & Load Schema", type="primary", use_container_width=True):
        st.session_state.last_error = ""
        st.session_state.query_result_df = None
        safe_dispose(st.session_state.engine)
        st.session_state.engine = None
        st.session_state.db_connected = False
        st.session_state.schema_text = ""
        st.session_state.schema_map = None
        st.session_state.sql_editor = ""

        try:
            with st.sidebar.status("Connecting database...", expanded=True) as status:
                status.write("connecting database...")
                engine = create_db_engine()
                status.write("Verifying connection...")
                ok, message = test_connection(engine)
                if not ok:
                    status.update(label="Connection failed", state="error")
                    st.session_state.last_error = message
                    st.sidebar.error(message)
                else:
                    status.write("detecting schema...")
                    st.session_state.engine = engine
                    st.session_state.db_connected = True
                    tables = get_table_names(engine)
                    schema_map = get_schema_details(engine)
                    st.session_state.schema_map = schema_map if schema_map else None
                    st.session_state.schema_text = build_schema_prompt_for_llm(engine)
                    status.update(label="Connection successful", state="complete")
                    st.sidebar.success(message)
                    st.toast("Schema loaded.")
                    if not tables and not schema_map:
                        st.sidebar.warning("No tables found yet.")
        except Exception as exc:
            err = str(exc)
            st.session_state.last_error = err
            st.session_state.db_connected = False
            st.sidebar.error(f"Could not connect: {err}")

    st.sidebar.divider()
    if st.session_state.get("setting_advanced_mode", False):
        st.sidebar.markdown("### 📋 Detected Schema")
        smap = st.session_state.schema_map
        if not smap:
            st.sidebar.info("Connect to load tables and columns.")
        else:
            for tname in sorted(smap.keys()):
                cols = smap[tname]
                col_lines = "\n".join(f"- `{c['name']}` — {_short_type_ui(c.get('type', ''))}" for c in cols)
                with st.sidebar.expander(f"**{tname}** ({len(cols)} cols)", expanded=False):
                    st.sidebar.markdown(col_lines)
        st.sidebar.divider()

    st.sidebar.divider()
    st.sidebar.markdown("### 🕒 Query History")
    hist: List[dict] = st.session_state.query_history
    if not hist:
        st.sidebar.caption("Run a successful query to build history.")
    else:
        for entry in hist:
            eid = entry["id"]
            ts: datetime = entry["timestamp"]
            preview = entry["sql"].replace("\n", " ").strip()
            if len(preview) > 100:
                preview = preview[:100] + "…"
            q_short = entry["question"][:120] + ("…" if len(entry["question"]) > 120 else "")
            st.sidebar.markdown(
                f'<div class="history-card">'
                f"<small style='color: #64748b;'>{ts:%Y-%m-%d %H:%M}</small><br/>"
                f"<strong style='color: #0f172a;'>Q:</strong> {html.escape(q_short)}<br/>"
                f"<strong style='color: #0f172a;'>Rows:</strong> {entry['row_count']}"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.session_state.get("setting_advanced_mode", False):
                st.sidebar.code(preview, language="sql")
            c1, c2 = st.sidebar.columns(2)
            c1.button(
                "Reuse Q",
                key=f"hq_rq_{eid}",
                use_container_width=True,
                on_click=_reuse_question_cb,
                args=(entry["question"],),
            )
            c2.button(
                "Reuse SQL",
                key=f"hq_rs_{eid}",
                use_container_width=True,
                on_click=_reuse_sql_cb,
                args=(entry["sql"],),
            )
            c3, c4 = st.sidebar.columns(2)
            c3.button(
                "View result",
                key=f"hq_vw_{eid}",
                use_container_width=True,
                on_click=_view_result_cb,
                args=(eid,),
            )
            c4.button(
                "Delete",
                key=f"hq_dl_{eid}",
                use_container_width=True,
                on_click=_delete_entry_cb,
                args=(eid,),
            )

    st.sidebar.divider()
    st.sidebar.markdown("### 🧹 Clear Actions")
    c5, c6 = st.sidebar.columns(2)
    c5.button(
        "Clear Result",
        key="sidebar_clear_result",
        use_container_width=True,
        on_click=_clear_result_cb,
    )
    c6.button(
        "Clear History",
        key="sidebar_clear_hist",
        use_container_width=True,
        on_click=_clear_history_cb,
    )
    st.sidebar.button(
        "Reset Session",
        key="sidebar_reset",
        use_container_width=True,
        on_click=_reset_app_cb,
        type="primary"
    )

    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Settings")
    st.session_state["setting_advanced_mode"] = st.sidebar.toggle("Advanced Mode", value=st.session_state.get("setting_advanced_mode", False), help="Show SQL, raw errors, and detected schema.")
    st.session_state["setting_auto_chart"] = st.sidebar.toggle("Auto Dashboard", value=st.session_state.get("setting_auto_chart", True), help="Automatically suggest charts based on results.")
    st.session_state["setting_history"] = st.sidebar.toggle("Save History", value=st.session_state.get("setting_history", True), help="Save queries to history.")
    st.session_state["setting_lang_assist"] = st.sidebar.toggle("Language Assist", value=st.session_state.get("setting_lang_assist", True), help="Understand broken English, Hindi, and Gujarati.")


def _render_progress_steps():
    """Renders a dynamic step progress indicator."""
    step = 0
    if _db_connected():
        step = 1
        if st.session_state.sql_editor:
            step = 2
            if st.session_state.query_result_df is not None:
                step = 4
                
    steps = ["1. Connect DB", "2. Generate SQL", "3. Run Query", "4. Visualize", "5. Export"]
    
    cols = st.columns(len(steps))
    for i, s in enumerate(steps):
        with cols[i]:
            if i < step:
                st.markdown(f"<div class='step-done'>✅ {s}</div>", unsafe_allow_html=True)
            elif i == step:
                st.markdown(f"<div class='step-active'>🔵 {s}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='step-pending'>⚪ {s}</div>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SQL Database Analysis Tool",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)
_inject_app_styles()
_init_session_state()

st.markdown(
    """
    <div class="app-header-wrap">
        <h1>📊 SQL Database Analysis Tool</h1>
        <p>Ask questions in plain English and analyze database insights instantly</p>
    </div>
    """,
    unsafe_allow_html=True,
)

_render_sidebar()
_render_progress_steps()

tab_workspace, tab_tips = st.tabs(["💡 Workspace", "ℹ️ Tips"])

with tab_tips:
    st.markdown(
        """
        **How to use**
        1. In the sidebar, click **Connect & Load Schema** (configure `.env` first).
        2. Ask a question in plain English and generate SQL.
        3. Review and edit SQL, then **Run Query** — only **SELECT** / **WITH … SELECT** is allowed.
        4. Explore results, charts, and exports. Past runs appear under **Query History**.

        **Safety**  
        This tool never runs writes. If the database returns an error, edit SQL and try again.
        """
    )

with tab_workspace:
    if not _db_connected():
        st.info("👈 Connect to your database from the **sidebar** to enable questions and SQL generation.")

    # ----- Step 2: Ask -----
    with st.container(border=True):
        st.subheader("🗣️ Step 2: Voice or Text Query")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**🎙️ Voice Query (Optional)**")
            st.caption("Click the microphone to speak your question.")
            if speech_to_text is not None:
                spoken_text = speech_to_text(
                    language='en',
                    use_container_width=True,
                    just_once=True,
                    key='STT'
                )
                if spoken_text:
                    st.session_state["user_query"] = spoken_text
                    st.success(f"Recognized: {spoken_text}")
            else:
                st.warning("Voice module not installed. Please run `pip install streamlit-mic-recorder`")

        with col2:
            st.text_area(
                "Describe what you need",
                placeholder='e.g. top 5 cities by sales, monthly revenue, customers with highest spending',
                height=120,
                key="user_query",
                help="Type or use Voice Input. Schema from the sidebar is sent to the model.",
            )
            
        gen_clicked = st.button("✨ Generate SQL", type="primary", key="btn_generate_sql")

    if gen_clicked:
        query = st.session_state["user_query"].strip()
        if not query:
            st.warning("Please type a question before generating SQL.")
        elif not _db_connected():
            st.error("Connect in the sidebar first so schema-aware SQL can be generated.")
        else:
            try:
                from llm_sql_generator import normalize_user_query, generate_sql_from_question
                with st.status("Understanding & generating...", expanded=True) as status:
                    if st.session_state.get("setting_lang_assist", True):
                        status.write("Normalizing language...")
                        normalized_q = normalize_user_query(query)
                    else:
                        normalized_q = query
                    st.session_state["normalized_query"] = normalized_q
                    
                    if st.session_state.get("setting_advanced_mode", False) and normalized_q != query:
                        st.info(f"**Normalized Query:** {normalized_q}")

                    status.write("Reading database schema...")
                    status.write("generating SQL...")
                    sql = generate_sql_from_question(
                        question=normalized_q,
                        schema_text=st.session_state.schema_text,
                    )
                    status.write("validating SQL...")
                    ok, reason = validate_read_only_sql(sql)
                    if not ok:
                        raise ValueError(reason)
                    st.session_state.sql_editor = sql
                    st.session_state.last_error = ""
                    status.update(label="SQL ready", state="complete")
                st.success("✅ Query understood and ready to run.")
                st.toast("SQL generated.")
            except ValueError as exc:
                msg = str(exc)
                st.session_state.last_error = msg
                if st.session_state.get("setting_advanced_mode", False):
                    st.error(msg)
                else:
                    st.error("I could not understand this request clearly. Please try again with simple terms like: top 5 cities by sales.")
            except RuntimeError as exc:
                st.session_state.last_error = str(exc)
                if st.session_state.get("setting_advanced_mode", False):
                    st.error(str(exc))
                else:
                    st.error("There was a problem generating the answer. Please try rephrasing your question.")
            except Exception as exc:
                st.session_state.last_error = str(exc)
                if st.session_state.get("setting_advanced_mode", False):
                    st.error(f"Unexpected error: {exc}")
                else:
                    st.error("An unexpected error occurred. Please try again.")

    st.markdown("<br/>", unsafe_allow_html=True)

    # ----- Step 3: SQL -----
    with st.container(border=True):
        if st.session_state.get("setting_advanced_mode", False):
            st.subheader("💻 Step 3: Generated SQL")
            st.info("Only **SELECT** queries are allowed for safety. You can edit the SQL before running.")
            sql_input = st.text_area(
                "SQL Editor",
                value=st.session_state.sql_editor,
                height=200,
                help="Single SELECT or WITH … SELECT. Re-validated on every run.",
            )
            if sql_input != st.session_state.sql_editor:
                st.session_state.sql_editor = sql_input
        else:
            st.subheader("💻 Step 3: Run Query")
            if st.session_state.sql_editor:
                st.success("Query securely prepared. Click Run to view results.")
                st.code(st.session_state.sql_editor, language="sql")
            else:
                st.info("Ask a question above to generate a query.")
                
        run_clicked = st.button("▶️ Run Query", type="primary", key="btn_run_query")

    if run_clicked:
        sql_query = (st.session_state.sql_editor or "").strip()
        if not _db_connected():
            st.error("Connect in the sidebar before running a query.")
        elif not sql_query:
            st.error("Add SQL to run, or generate it from Step 2.")
        else:
            ok_pre, reason_pre = validate_read_only_sql(sql_query)
            if not ok_pre:
                st.error(f"**Invalid or unsafe SQL.** {reason_pre}")
            else:
                try:
                    try:
                        settings = Settings()
                        settings.validate()
                        sql_dialect = (
                            "postgres" if settings.db_type == "postgresql" else "mysql"
                        )
                    except Exception:
                        sql_dialect = "mysql"

                    with st.status("running query...", expanded=True) as status:
                        status.write("running query...")
                        result_df, final_sql, fix_messages = execute_sql_query_with_fixes(
                            engine=st.session_state.engine,
                            sql_query=sql_query,
                            schema_text=st.session_state.schema_text or "",
                            schema_details=st.session_state.schema_map,
                            question=st.session_state["user_query"],
                            sql_dialect=sql_dialect,
                            max_llm_fixes=3,
                        )
                        status.write("Fetching results...")
                        for msg in fix_messages:
                            if st.session_state.get("setting_advanced_mode", False):
                                st.info(msg)
                        if final_sql.strip() != sql_query.strip() and st.session_state.get("setting_advanced_mode", False):
                            st.session_state.sql_editor = final_sql
                            st.success("SQL was auto-corrected to match your schema.")
                        
                        status.write("Preparing insights...")
                        st.session_state.last_error = ""
                        st.session_state.query_result_df = result_df
                        
                        from llm_sql_generator import generate_insight_summary
                        insight = generate_insight_summary(st.session_state["user_query"], result_df)
                        st.session_state["insight_summary"] = insight
                        
                        if st.session_state.get("setting_history", True):
                            _append_query_history(st.session_state["user_query"], final_sql, result_df, insight)
                        
                        status.update(label="Query finished", state="complete")
                    st.toast("Results updated.")
                except ValueError as exc:
                    st.session_state.last_error = str(exc)
                    if st.session_state.get("setting_advanced_mode", False):
                        st.error(f"**Invalid SQL.**\n\n{exc}")
                    else:
                        st.error("This question requires data that might not exist in the database. Please try a different question.")
                except RuntimeError as exc:
                    st.session_state.last_error = str(exc)
                    if st.session_state.get("setting_advanced_mode", False):
                        st.error(f"**Query failed.**\n\n{exc}")
                    else:
                        st.error("Unable to execute query. Please try rephrasing your question.")
                except Exception as exc:
                    st.session_state.last_error = str(exc)
                    if st.session_state.get("setting_advanced_mode", False):
                        st.error(f"**Unexpected error.**\n\n{exc}")
                    else:
                        st.error("An unexpected error occurred while fetching data.")

    # ----- Steps 4–6: Results, chart, export -----
    if st.session_state.query_result_df is not None:
        st.markdown("<br/>", unsafe_allow_html=True)
        result_df = st.session_state.query_result_df

        with st.container(border=True):
            st.subheader("📋 Step 4: Results")
            if st.session_state.get("insight_summary"):
                st.success(f"💡 **Insight:** {st.session_state['insight_summary']}")

            if result_df.empty:
                st.warning("The query returned **no rows**.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", len(result_df))
                col2.metric("Columns", len(result_df.columns))
                col3.metric("Last Run", st.session_state.last_export_ts)

                st.dataframe(
                    _dataframe_display_copy(result_df),
                    use_container_width=True,
                    hide_index=True,
                )
                if st.session_state.get("setting_advanced_mode", False):
                    _render_numeric_summary(result_df)

        st.markdown("<br/>", unsafe_allow_html=True)
        
        # Prepare charts dataframe and get columns
        chart_df = prepare_chart_dataframe(result_df)
        numeric_cols, categorical_cols, datetime_cols = get_column_types(result_df)
        
        with st.expander("🛠️ Debug: Detected Column Types", expanded=False):
            st.write("**Numeric columns:**", numeric_cols)
            st.write("**Categorical columns:**", categorical_cols)
            st.write("**Date columns:**", datetime_cols)

        st.markdown("<br/>", unsafe_allow_html=True)
        
        if st.session_state.get("setting_auto_chart", True):
            with st.container(border=True):
                st.subheader("📈 Auto Dashboard")
                render_auto_dashboard(chart_df, numeric_cols, categorical_cols, datetime_cols)
            st.markdown("<br/>", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("🎨 Step 5: Manual Charts")
            render_chart_section(chart_df, numeric_cols, categorical_cols, datetime_cols)
            _sync_history_chart_from_session()

        st.markdown("<br/>", unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("📥 Step 6: Export")
            st.caption("Export the raw data (original column names).")
            render_export_buttons(result_df)
    else:
        st.caption("Results appear here after a successful query (previous results stay until cleared).")
