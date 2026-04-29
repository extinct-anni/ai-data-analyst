"""
viz.py — Chart generation helpers for Streamlit
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# ─── Style ────────────────────────────────────────────────────────────────────

PALETTE = "mako"
FIG_BG = "#0f1117"
AXES_BG = "#1a1d27"
TEXT_COLOR = "#e0e0e0"
ACCENT = "#4f9cf9"


def _style_fig(fig, ax):
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")
    return fig, ax


# ─── Individual Chart Functions ───────────────────────────────────────────────

def plot_histogram(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[col].dropna(), bins=30, color=ACCENT, edgecolor="#0f1117", alpha=0.85)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    _style_fig(fig, ax)
    st.pyplot(fig)
    plt.close(fig)


def plot_bar(df: pd.DataFrame, x: str, y: str = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    if y:
        data = df.groupby(x)[y].mean().sort_values(ascending=False).head(20)
        ax.bar(data.index.astype(str), data.values, color=ACCENT, edgecolor="#0f1117")
        ax.set_ylabel(f"Mean {y}")
        ax.set_title(f"{y} by {x}")
    else:
        data = df[x].value_counts().head(20)
        ax.bar(data.index.astype(str), data.values, color=ACCENT, edgecolor="#0f1117")
        ax.set_ylabel("Count")
        ax.set_title(f"Count of {x}")
    plt.xticks(rotation=45, ha="right", color=TEXT_COLOR)
    _style_fig(fig, ax)
    st.pyplot(fig)
    plt.close(fig)


def plot_line(df: pd.DataFrame, x: str, y: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    df_sorted = df[[x, y]].dropna().sort_values(x)
    ax.plot(df_sorted[x], df_sorted[y], color=ACCENT, linewidth=2)
    ax.fill_between(df_sorted[x], df_sorted[y], alpha=0.15, color=ACCENT)
    ax.set_title(f"{y} over {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    _style_fig(fig, ax)
    st.pyplot(fig)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x: str, y: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[x], df[y], alpha=0.5, color=ACCENT, s=20, edgecolors="none")
    # Add trend line
    try:
        m, b = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
        xs = np.linspace(df[x].min(), df[x].max(), 100)
        ax.plot(xs, m * xs + b, color="#ff6b6b", linewidth=1.5, linestyle="--")
    except Exception:
        pass
    ax.set_title(f"{x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    _style_fig(fig, ax)
    st.pyplot(fig)
    plt.close(fig)


def plot_box(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(
        df[col].dropna(),
        patch_artist=True,
        boxprops=dict(facecolor=ACCENT, color=TEXT_COLOR),
        medianprops=dict(color="#ff6b6b", linewidth=2),
        whiskerprops=dict(color=TEXT_COLOR),
        capprops=dict(color=TEXT_COLOR),
        flierprops=dict(marker="o", color="#ff6b6b", markersize=4),
    )
    ax.set_title(f"Box Plot — {col}")
    ax.set_ylabel(col)
    _style_fig(fig, ax)
    st.pyplot(fig)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=["int64", "float64"])
    if num_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for a correlation heatmap.")
        return
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr)), max(5, len(corr) - 1)))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 9},
    )
    ax.set_title("Correlation Matrix")
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=TEXT_COLOR)
    st.pyplot(fig)
    plt.close(fig)


def plot_pie(df: pd.DataFrame, col: str):
    data = df[col].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = sns.color_palette("mako", len(data))
    wedges, texts, autotexts = ax.pie(
        data.values,
        labels=data.index.astype(str),
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.82,
    )
    for t in texts + autotexts:
        t.set_color(TEXT_COLOR)
    ax.set_title(f"Distribution of {col}", color=TEXT_COLOR)
    fig.patch.set_facecolor(FIG_BG)
    st.pyplot(fig)
    plt.close(fig)


# ─── AI-Driven Auto Charts ────────────────────────────────────────────────────

def render_ai_charts(df: pd.DataFrame, suggestions_json: str):
    """
    Render charts based on AI suggestions JSON.
    suggestions_json: JSON array from ai_engine.suggest_charts()
    """
    try:
        suggestions = json.loads(suggestions_json)
    except Exception:
        st.warning("Could not parse chart suggestions.")
        return

    for i, s in enumerate(suggestions[:3]):
        chart_type = s.get("chart_type", "")
        x = s.get("x")
        y = s.get("y")
        reason = s.get("reason", "")

        if reason:
            st.caption(f"💡 {reason}")

        try:
            if chart_type == "histogram" and x in df.columns:
                plot_histogram(df, x)
            elif chart_type == "bar" and x in df.columns:
                plot_bar(df, x, y if y and y in df.columns else None)
            elif chart_type == "line" and x in df.columns and y in df.columns:
                plot_line(df, x, y)
            elif chart_type == "scatter" and x in df.columns and y in df.columns:
                plot_scatter(df, x, y)
            elif chart_type == "box" and x in df.columns:
                plot_box(df, x)
            elif chart_type == "heatmap":
                plot_heatmap(df)
            elif chart_type == "pie" and x in df.columns:
                plot_pie(df, x)
        except Exception as e:
            st.warning(f"Could not render {chart_type} chart: {e}")
