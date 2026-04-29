"""
app.py — AI-Powered Autonomous Data Analyst
Run: streamlit run app.py
"""

import os
import sys
import json
import warnings
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# ─── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --bg: #0f1117;
        --surface: #1a1d27;
        --border: #2a2d3a;
        --accent: #4f9cf9;
        --accent2: #a78bfa;
        --success: #34d399;
        --warn: #fbbf24;
        --danger: #f87171;
        --text: #e2e8f0;
        --muted: #64748b;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }

    .stButton > button {
        background: var(--accent) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-card .val {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
        font-family: 'Space Mono', monospace;
    }
    .metric-card .lbl {
        font-size: 0.75rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .insight-box {
        background: var(--surface);
        border-left: 3px solid var(--accent);
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        font-size: 0.93rem;
        line-height: 1.65;
    }

    .tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px;
        font-family: 'Space Mono', monospace;
    }
    .tag-num  { background: #1e3a5f; color: var(--accent);  border: 1px solid #2563eb44; }
    .tag-cat  { background: #2d1f5f; color: var(--accent2); border: 1px solid #7c3aed44; }
    .tag-dt   { background: #1a3a2f; color: var(--success); border: 1px solid #05966944; }

    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--muted);
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
    }

    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stTextInput > div > input { background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }
    .stSelectbox > div { background: var(--surface) !important; }
    div[data-testid="stExpander"] { background: var(--surface); border: 1px solid var(--border) !important; border-radius: 8px; }
    .stAlert { border-radius: 8px; }

    h1, h2, h3 { color: var(--text) !important; }
    h1 { font-family: 'Space Mono', monospace; font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── Imports (after path setup) ───────────────────────────────────────────────
from utils.schema import detect_schema, get_summary_stats
from utils.cleaning import (
    get_missing_report, fill_missing,
    get_outlier_report, remove_outliers, remove_duplicates, fix_dtypes
)
from utils.sql_engine import load_dataframe, execute_sql
from utils.viz import (
    plot_histogram, plot_bar, plot_line, plot_scatter,
    plot_box, plot_heatmap, plot_pie, render_ai_charts
)


# ─── State Helpers ────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "df": None,
        "df_original": None,
        "schema": None,
        "api_key_set": False,
        "db_loaded": False,
        "chat_history": [],
        "sql_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def set_df(df: pd.DataFrame):
    st.session_state.df = df
    st.session_state.schema = detect_schema(df)


init_state()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("# 🧠 AI Analyst")
#     st.markdown("---")

#     # API Key
#     st.markdown('<div class="section-header">Groq API Key</div>', unsafe_allow_html=True)
#     # Auto-load from secrets.toml or environment variable first
# groq_key = (
#     st.secrets.get("GROQ_API_KEY")          # ← from .streamlit/secrets.toml
#     or os.environ.get("GROQ_API_KEY")        # ← from system environment
# )

# # api_key_input = st.text_input(
# #     "API Key (optional — auto-loaded if set in secrets)",
# #     type="password",
# #     placeholder="gsk_...",
# #     key="api_key_input"
# # )

# # UI input overrides secrets if provided
# final_key = groq_key

# if final_key:
#     os.environ["GROQ_API_KEY"] = final_key
#     st.success("✓ API Key auto-loaded")
# else:
#     st.warning("No API key found")

#     st.markdown("---")

#     # Upload
#     st.markdown('<div class="section-header">Upload Data</div>', unsafe_allow_html=True)
#     uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
#     if uploaded:
#         try:
#             if uploaded.name.endswith(".csv"):
#                 df_raw = pd.read_csv(uploaded)
#             else:
#                 df_raw = pd.read_excel(uploaded)
#             st.session_state.df_original = df_raw.copy()
#             set_df(df_raw)
#             # Load into SQLite
#             os.makedirs("data", exist_ok=True)
#             load_dataframe(df_raw, "data/analyst.db")
#             st.session_state.db_loaded = True
#             st.success(f"✓ Loaded {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
#         except Exception as e:
#             st.error(f"Error loading file: {e}")

#     # Quick nav
#     if st.session_state.df is not None:
#         st.markdown("---")
#         st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
#         page = st.radio(
#             "Go to",
#             ["📊 Overview", "🧹 Clean Data", "📈 Visualize", "💬 Ask AI", "🔍 SQL Query"],
#             label_visibility="collapsed",
#         )
#     else:
#         page = "📊 Overview"

#     st.markdown("---")
#     st.markdown('<div style="color:#475569;font-size:0.7rem;text-align:center">Built with Streamlit + OpenAI</div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("# 🧠 AI Analyst")
    st.markdown("---")

    # ── Auto-load API key silently (no UI input) ──
    groq_key = (
        st.secrets.get("GROQ_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.session_state.api_key_set = True

    st.markdown("---")

    # ── Upload ────────────────────────────────────
    st.markdown('<div class="section-header">Upload Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
            st.session_state.df_original = df_raw.copy()
            set_df(df_raw)
            os.makedirs("data", exist_ok=True)
            load_dataframe(df_raw, "data/analyst.db")
            st.session_state.db_loaded = True
            st.success(f"✓ Loaded {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    # ── Navigation ────────────────────────────────
    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "Go to",
            ["📊 Overview", "🧹 Clean Data", "📈 Visualize", "💬 Ask AI", "🔍 SQL Query"],
            label_visibility="collapsed",
        )
    else:
        page = "📊 Overview"

    st.markdown("---")
    st.markdown(
        '<div style="color:#475569;font-size:0.7rem;text-align:center">Built with Streamlit + Grolag</div>',
        unsafe_allow_html=True
    )

# ─── No file uploaded ─────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("## 🧠 AI-Powered Data Analyst")
    st.markdown("Upload a CSV or Excel file in the sidebar to get started.")
    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in [
        (col1, "📁", "Upload", "CSV or Excel"),
        (col2, "🧹", "Clean", "Auto-clean your data"),
        (col3, "💬", "Ask AI", "Natural language insights"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-weight:600;margin:0.4rem 0">{title}</div>
                <div style="color:#64748b;font-size:0.83rem">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.stop()


df = st.session_state.df
schema = st.session_state.schema


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("## 📊 Dataset Overview")

    # Metrics
    rows, cols = schema["shape"]
    missing_total = sum(v["null_count"] for v in schema["profile"].values())
    dup_count = df.duplicated().sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    for c, val, lbl in [
        (c1, f"{rows:,}", "Rows"),
        (c2, str(cols), "Columns"),
        (c3, str(len(schema["numeric"])), "Numeric"),
        (c4, str(len(schema["categorical"])), "Categorical"),
        (c5, str(missing_total), "Missing Values"),
    ]:
        with c:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Column Tags
    st.markdown("**Column Types**")
    tag_html = ""
    for col in schema["numeric"]:
        tag_html += f'<span class="tag tag-num">📊 {col}</span>'
    for col in schema["categorical"]:
        tag_html += f'<span class="tag tag-cat">🏷 {col}</span>'
    for col in schema["datetime"]:
        tag_html += f'<span class="tag tag-dt">📅 {col}</span>'
    st.markdown(tag_html, unsafe_allow_html=True)

    st.markdown("---")

    # Data Preview
    st.markdown("**Data Preview**")
    st.dataframe(df.head(50), use_container_width=True)

    # Stats
    with st.expander("📐 Descriptive Statistics"):
        stats = get_summary_stats(df)
        if not stats.empty:
            st.dataframe(stats, use_container_width=True)
        else:
            st.info("No numeric columns found.")

    # AI Summary
    if st.session_state.api_key_set:
        if st.button("🤖 Generate AI Dataset Summary"):
            with st.spinner("Generating summary..."):
                try:
                    from utils.ai_engine import generate_dataset_summary
                    summary = generate_dataset_summary(df, schema)
                    st.markdown(f'<div class="insight-box">{summary}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI error: {e}")
    else:
        st.info("🔑 Add your OpenAI API key in the sidebar to enable AI features.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CLEAN DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧹 Clean Data":
    st.markdown("## 🧹 Data Cleaning Engine")

    tab1, tab2, tab3, tab4 = st.tabs(["Missing Values", "Outliers", "Duplicates", "AI Suggestions"])

    # ── Tab 1: Missing Values
    with tab1:
        missing_report = get_missing_report(df)
        if missing_report.empty:
            st.success("✅ No missing values found!")
        else:
            st.warning(f"Found missing values in {len(missing_report)} column(s).")
            st.dataframe(missing_report, use_container_width=True)

            strategy = st.selectbox(
                "Fill Strategy",
                ["mean", "median", "mode", "drop"],
                help="For numeric cols: mean/median/mode fills. For categorical: mode fills. 'drop' removes rows."
            )
            if st.button("🧹 Apply Cleaning"):
                cleaned, actions = fill_missing(df, strategy)
                set_df(cleaned)
                load_dataframe(cleaned, "data/analyst.db")
                st.success(f"Cleaned! {len(actions)} actions taken.")
                for a in actions:
                    st.markdown(f"  - {a}")
                st.rerun()

    # ── Tab 2: Outliers
    with tab2:
        outlier_report = get_outlier_report(df)
        if outlier_report.empty:
            st.success("✅ No significant outliers found.")
        else:
            st.warning(f"Outliers detected in {len(outlier_report)} column(s).")
            st.dataframe(outlier_report, use_container_width=True)

            num_cols = schema["numeric"]
            if num_cols:
                col_to_clean = st.selectbox("Column to clean", num_cols)
                method = st.radio("Detection method", ["iqr", "zscore"], horizontal=True)
                if st.button("Remove Outliers"):
                    cleaned, removed = remove_outliers(df, col_to_clean, method)
                    set_df(cleaned)
                    load_dataframe(cleaned, "data/analyst.db")
                    st.success(f"Removed {removed} outlier rows from '{col_to_clean}'.")
                    st.rerun()

    # ── Tab 3: Duplicates
    with tab3:
        dup_count = df.duplicated().sum()
        if dup_count == 0:
            st.success("✅ No duplicate rows found.")
        else:
            st.warning(f"Found {dup_count} duplicate rows ({dup_count/len(df)*100:.1f}% of data).")
            st.dataframe(df[df.duplicated()].head(20), use_container_width=True)
            if st.button("Remove Duplicates"):
                cleaned, removed = remove_duplicates(df)
                set_df(cleaned)
                load_dataframe(cleaned, "data/analyst.db")
                st.success(f"Removed {removed} duplicate rows.")
                st.rerun()

        st.markdown("---")
        if st.button("🔧 Auto-Fix Data Types"):
            fixed, actions = fix_dtypes(df)
            if actions:
                set_df(fixed)
                st.success("Fixed column types:")
                for a in actions:
                    st.markdown(f"  - {a}")
            else:
                st.info("All columns already have correct types.")

    # ── Tab 4: AI Suggestions
    with tab4:
        if not st.session_state.api_key_set:
            st.info("🔑 Add your OpenAI API key in the sidebar.")
        else:
            if st.button("🤖 Get AI Cleaning Recommendations"):
                with st.spinner("Analyzing data quality..."):
                    try:
                        from utils.ai_engine import suggest_cleaning
                        missing_report = get_missing_report(df)
                        outlier_report = get_outlier_report(df)
                        suggestions = suggest_cleaning(missing_report, outlier_report, schema)
                        st.markdown(f'<div class="insight-box">{suggestions}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"AI error: {e}")

    # Reset
    st.markdown("---")
    if st.session_state.df_original is not None:
        if st.button("↩️ Reset to Original Data"):
            set_df(st.session_state.df_original.copy())
            load_dataframe(st.session_state.df_original.copy(), "data/analyst.db")
            st.success("Reset to original dataset.")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualize":
    st.markdown("## 📈 Visualization Engine")

    tab1, tab2, tab3 = st.tabs(["Manual Charts", "Correlation Heatmap", "AI Auto-Charts"])

    num_cols = schema["numeric"]
    cat_cols = schema["categorical"]

    # ── Manual Charts
    with tab1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Histogram", "Bar Chart", "Scatter Plot", "Line Chart", "Box Plot", "Pie Chart"]
        )

        if chart_type == "Histogram":
            if num_cols:
                col = st.selectbox("Column", num_cols)
                plot_histogram(df, col)
            else:
                st.warning("No numeric columns.")

        elif chart_type == "Bar Chart":
            if cat_cols:
                x = st.selectbox("Category (X)", cat_cols)
                y = st.selectbox("Value (Y, optional)", ["None"] + num_cols)
                plot_bar(df, x, None if y == "None" else y)
            else:
                st.warning("No categorical columns.")

        elif chart_type == "Scatter Plot":
            if len(num_cols) >= 2:
                x = st.selectbox("X Axis", num_cols, index=0)
                y = st.selectbox("Y Axis", num_cols, index=1)
                plot_scatter(df, x, y)
            else:
                st.warning("Need at least 2 numeric columns.")

        elif chart_type == "Line Chart":
            all_cols = df.columns.tolist()
            x = st.selectbox("X Axis", all_cols)
            y = st.selectbox("Y Axis", num_cols if num_cols else all_cols)
            plot_line(df, x, y)

        elif chart_type == "Box Plot":
            if num_cols:
                col = st.selectbox("Column", num_cols)
                plot_box(df, col)
            else:
                st.warning("No numeric columns.")

        elif chart_type == "Pie Chart":
            if cat_cols:
                col = st.selectbox("Column", cat_cols)
                plot_pie(df, col)
            else:
                st.warning("No categorical columns.")

    # ── Heatmap
    with tab2:
        st.markdown("Correlation matrix for all numeric columns.")
        if len(num_cols) >= 2:
            plot_heatmap(df)
        else:
            st.warning("Need at least 2 numeric columns.")

    # ── AI Auto-Charts
    with tab3:
        if not st.session_state.api_key_set:
            st.info("🔑 Add your OpenAI API key in the sidebar.")
        else:
            question_hint = st.text_input(
                "Optional: what are you looking for?",
                placeholder="e.g. sales trends, top categories..."
            )
            if st.button("🤖 Generate Smart Charts"):
                with st.spinner("AI choosing best charts..."):
                    try:
                        from utils.ai_engine import suggest_charts
                        suggestions_json = suggest_charts(schema, question_hint)
                        render_ai_charts(df, suggestions_json)
                    except Exception as e:
                        st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ASK AI
# ═══════════════════════════════════════════════════════════════════════════════
# elif page == "💬 Ask AI":
#     st.markdown("## 💬 AI Insight Generator")

#     if not st.session_state.api_key_set:
#         st.info("🔑 Add your OpenAI API key in the sidebar to use this feature.")
#         st.stop()

#     from utils.ai_engine import get_insight, suggest_charts
#     from utils.viz import render_ai_charts

#     # Chat history display
#     for msg in st.session_state.chat_history:
#         role = msg["role"]
#         content = msg["content"]
#         if role == "user":
#             st.markdown(f"""
#             <div style="text-align:right;margin:0.5rem 0">
#                 <span style="background:#2563eb;color:#fff;padding:0.5rem 1rem;border-radius:16px 16px 4px 16px;font-size:0.88rem;display:inline-block;max-width:75%">{content}</span>
#             </div>""", unsafe_allow_html=True)
#         else:
#             st.markdown(f'<div class="insight-box">{content}</div>', unsafe_allow_html=True)

#     # Question shortcuts
#     st.markdown("**Quick questions:**")
#     shortcuts = [
#         "What are the key trends in this data?",
#         "Which columns are most correlated?",
#         "What are the top performing categories?",
#         "Are there any anomalies or outliers I should know about?",
#     ]
#     cols = st.columns(2)
#     for i, q in enumerate(shortcuts):
#         with cols[i % 2]:
#             if st.button(q, key=f"shortcut_{i}", use_container_width=True):
#                 st.session_state._pending_question = q

#     # Input
#     question = st.text_input(
#         "Ask anything about your data",
#         placeholder="e.g. Why are sales dropping? What's the average revenue by category?",
#         key="ai_question",
#     )

#     # Handle shortcut press
#     if hasattr(st.session_state, "_pending_question"):
#         question = st.session_state._pending_question
#         del st.session_state._pending_question

#     if st.button("Send 🚀") or question:
#         if question:
#             st.session_state.chat_history.append({"role": "user", "content": question})

#             with st.spinner("Thinking..."):
#                 try:
#                     stats = get_summary_stats(df)
#                     stats_str = stats.to_string() if not stats.empty else ""
#                     insight = get_insight(question, df, schema, stats_str)
#                     st.session_state.chat_history.append({"role": "assistant", "content": insight})

#                     # Auto-chart if question implies visuals
#                     visual_keywords = ["trend", "chart", "plot", "visuali", "show", "compare", "distribut"]
#                     if any(kw in question.lower() for kw in visual_keywords):
#                         suggestions_json = suggest_charts(schema, question)
#                         render_ai_charts(df, suggestions_json)

#                 except Exception as e:
#                     st.error(f"AI error: {e}")
#             st.rerun()

#     if st.session_state.chat_history:
#         if st.button("🗑 Clear Chat"):
#             st.session_state.chat_history = []
#             st.rerun()


elif page == "💬 Ask AI":
    st.markdown("## 💬 AI Insight Generator")

    if not st.session_state.api_key_set:
        st.info("🔑 Add your API key in secrets.toml")
        st.stop()

    from utils.ai_engine import get_insight, suggest_charts
    from utils.viz import render_ai_charts

    # ── Display chat history ──────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="text-align:right;margin:0.5rem 0">
                <span style="background:#2563eb;color:#fff;padding:0.5rem 1rem;
                border-radius:16px 16px 4px 16px;font-size:0.88rem;
                display:inline-block;max-width:75%">{msg["content"]}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="insight-box">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

    # ── Quick shortcut buttons ────────────────────
    st.markdown("**Quick questions:**")
    shortcuts = [
        "What are the key trends in this data?",
        "Which columns are most correlated?",
        "What are the top performing categories?",
        "Are there any anomalies I should know about?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(shortcuts):
        with cols[i % 2]:
            if st.button(q, key=f"shortcut_{i}", use_container_width=True):
                # Only process if not already last user message
                last_user = next(
                    (m["content"] for m in reversed(st.session_state.chat_history)
                     if m["role"] == "user"), None
                )
                if last_user != q:
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    with st.spinner("Thinking..."):
                        try:
                            from utils.schema import get_summary_stats
                            stats = get_summary_stats(df)
                            stats_str = stats.to_string() if not stats.empty else ""
                            insight = get_insight(q, df, schema, stats_str)
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": insight}
                            )
                        except Exception as e:
                            st.error(f"AI error: {e}")
                    st.rerun()

    # ── Text input ────────────────────────────────
    question = st.chat_input("Ask anything about your data...")

    if question:
        # Check it's not a duplicate of last message
        last_user = next(
            (m["content"] for m in reversed(st.session_state.chat_history)
             if m["role"] == "user"), None
        )
        if last_user != question:
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Thinking..."):
                try:
                    from utils.schema import get_summary_stats
                    stats = get_summary_stats(df)
                    stats_str = stats.to_string() if not stats.empty else ""
                    insight = get_insight(question, df, schema, stats_str)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": insight}
                    )

                    # Auto chart for visual questions
                    visual_keywords = ["trend", "chart", "plot", "show", "compare", "distribut"]
                    if any(kw in question.lower() for kw in visual_keywords):
                        suggestions_json = suggest_charts(schema, question)
                        render_ai_charts(df, suggestions_json)

                except Exception as e:
                    st.error(f"AI error: {e}")
            st.rerun()

    # ── Clear chat ────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SQL QUERY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 SQL Query":
    st.markdown("## 🔍 Natural Language → SQL Engine")

    if not st.session_state.db_loaded:
        st.warning("Database not loaded. Please re-upload your file.")
        st.stop()

    tab1, tab2 = st.tabs(["Ask in English", "Write SQL Directly"])

    # ── NL to SQL
    with tab1:
        if not st.session_state.api_key_set:
            st.info("🔑 Add your OpenAI API key for NL → SQL conversion.")
        else:
            from utils.ai_engine import nl_to_sql, explain_result

            nl_question = st.text_input(
                "Ask a question (will be converted to SQL)",
                placeholder="e.g. What are the top 10 customers by revenue?"
            )

            # SQL history shortcuts
            if st.session_state.sql_history:
                with st.expander("📜 Previous queries"):
                    for h in st.session_state.sql_history[-5:]:
                        st.code(h["sql"], language="sql")
                        if h.get("result_shape"):
                            st.caption(f"→ {h['result_shape']}")

            if st.button("Convert & Run ⚡"):
                if nl_question:
                    with st.spinner("Converting to SQL..."):
                        try:
                            sample_rows = df.head(3).to_dict(orient="records")
                            sql = nl_to_sql(nl_question, schema["columns"], sample_rows)

                            st.markdown("**Generated SQL:**")
                            st.code(sql, language="sql")

                            result = execute_sql(sql)
                            st.markdown(f"**Result** — {len(result):,} rows × {len(result.columns)} cols")
                            st.dataframe(result, use_container_width=True)

                            # Store history
                            st.session_state.sql_history.append({
                                "question": nl_question,
                                "sql": sql,
                                "result_shape": f"{result.shape[0]} rows × {result.shape[1]} cols"
                            })

                            # Explain result
                            if not result.empty:
                                with st.spinner("Generating business explanation..."):
                                    explanation = explain_result(result, nl_question)
                                st.markdown("**💡 Business Insight:**")
                                st.markdown(f'<div class="insight-box">{explanation}</div>', unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Error: {e}")

    # ── Manual SQL
    with tab2:
        st.markdown(f"**Table name:** `data`   |   **Columns:** `{', '.join(schema['columns'])}`")

        manual_sql = st.text_area(
            "Write SQL",
            placeholder="SELECT * FROM data LIMIT 10;",
            height=140,
        )

        if st.button("▶ Run SQL"):
            if manual_sql.strip():
                try:
                    result = execute_sql(manual_sql.strip().rstrip(";"))
                    st.success(f"✓ {len(result):,} rows returned")
                    st.dataframe(result, use_container_width=True)

                    # Quick stats on numeric result cols
                    num_result_cols = result.select_dtypes(include=["number"]).columns
                    if not num_result_cols.empty:
                        with st.expander("📐 Result Stats"):
                            st.dataframe(result[num_result_cols].describe().round(2), use_container_width=True)

                except Exception as e:
                    st.error(f"SQL Error: {e}")
