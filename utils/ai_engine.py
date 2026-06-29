"""
ai_engine.py — All AI API calls using Groq (free)
"""

import json
import os
import re
import pandas as pd
from groq import Groq
from typing import Optional


# ─── Core Client ──────────────────────────────────────────────────────────────

def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Add it to .streamlit/secrets.toml")
    return Groq(api_key=api_key)


def _chat(system: str, user: str, max_tokens: int = 800, **kwargs) -> str:
    """Core wrapper — all AI calls go through here."""
    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


# ─── 1. Data Insight Generator ────────────────────────────────────────────────

def get_insight(
    question: str,
    df: pd.DataFrame,
    schema: dict,
    stats_summary: Optional[str] = None,
) -> str:
    system = """You are a senior data analyst who explains findings clearly 
to non-technical business users. Be concise, actionable, and use bullet 
points for clarity. Highlight the most important numbers. Avoid jargon."""

    col_list = ", ".join(schema["columns"])
    num_cols  = ", ".join(schema["numeric"])      or "None"
    cat_cols  = ", ".join(schema["categorical"])  or "None"
    rows, cols = schema["shape"]
    stats_block = f"\nDescriptive Stats:\n{stats_summary}" if stats_summary else ""

    user = f"""Dataset: {rows} rows x {cols} columns
Numeric columns: {num_cols}
Categorical columns: {cat_cols}
All columns: {col_list}
{stats_block}

Business question: {question}

Provide a clear, insightful answer. Include specific numbers where relevant."""

    return _chat(system, user, max_tokens=600)


# ─── 2. NL → SQL ──────────────────────────────────────────────────────────────

def nl_to_sql(question: str, columns: list, sample_rows: list) -> str:
    from utils.sql_engine import build_nl_to_sql_prompt
    prompt = build_nl_to_sql_prompt(question, columns, sample_rows)
    system = "You are an expert SQL generator. Output ONLY raw SQL, nothing else. No markdown, no explanation."
    return _chat(system, prompt, max_tokens=300)


# ─── 3. Result Explainer ──────────────────────────────────────────────────────

def explain_result(result_df: pd.DataFrame, question: str) -> str:
    system = """You are a business intelligence analyst.
Explain data results in plain English. Be specific about what the numbers 
mean for the business. Keep it under 150 words."""

    preview = result_df.head(10).to_string(index=False)
    user = f"""Original question: {question}

Query result:
{preview}

Explain what this result means in business terms."""

    return _chat(system, user, max_tokens=300)


# ─── 4. Cleaning Suggestions ─────────────────────────────────────────────────

def suggest_cleaning(
    missing_report: pd.DataFrame,
    outlier_report: pd.DataFrame,
    schema: dict
) -> str:
    system = """You are a data quality expert.
Given missing values and outlier stats, recommend the best cleaning strategy.
Be specific: name columns and recommend actions. Use bullet points."""

    missing_str = missing_report.to_string() if not missing_report.empty else "No missing values."
    outlier_str = outlier_report.to_string() if not outlier_report.empty else "No significant outliers."

    user = f"""Dataset shape: {schema['shape'][0]} rows x {schema['shape'][1]} cols
Column types — Numeric: {schema['numeric']}, Categorical: {schema['categorical']}

Missing values:
{missing_str}

Outliers (IQR method):
{outlier_str}

Provide specific cleaning recommendations for each problem column."""

    return _chat(system, user, max_tokens=500)


# ─── 5. Auto Chart Suggester ─────────────────────────────────────────────────

def suggest_charts(schema: dict, question: str = "") -> str:
    system = """You are a data visualization expert.
Suggest the 3 most useful charts for this dataset.
Return ONLY a JSON array. Each item must have:
{"chart_type": "...", "x": "col_name", "y": "col_name_or_null", "reason": "..."}
chart_type options: histogram, bar, line, scatter, box, heatmap, pie.
Return ONLY the JSON array, no explanation, no markdown."""

    user = f"""Numeric columns: {schema['numeric']}
Categorical columns: {schema['categorical']}
Datetime columns: {schema['datetime']}
User question (if any): {question}

Suggest 3 charts. Return only JSON array."""

    result = _chat(system, user, max_tokens=400)

    # Strip markdown fences if model adds them
    result = re.sub(r"```(?:json)?", "", result).strip().rstrip("`").strip()

    try:
        json.loads(result)  # validate
    except Exception:
        result = "[]"
    return result


# ─── 6. Dataset Summary ──────────────────────────────────────────────────────

def generate_dataset_summary(df: pd.DataFrame, schema: dict) -> str:
    system = """You are a data analyst. Write a concise 3-sentence executive 
summary of this dataset for a business user."""

    rows, cols = schema["shape"]
    user = f"""Dataset: {rows} rows, {cols} columns
Numeric: {schema['numeric']}
Categorical: {schema['categorical']}
Datetime: {schema['datetime']}
Sample (first 3 rows):
{df.head(3).to_string(index=False)}

Write a 3-sentence executive summary."""

    return _chat(system, user, max_tokens=200)
