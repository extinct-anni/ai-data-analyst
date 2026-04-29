"""
sql_engine.py — Load DataFrame into SQLite, convert NL → SQL, execute queries
"""

import re
import pandas as pd
from sqlalchemy import create_engine, text


_engine = None
_table_name = "data"


def load_dataframe(df: pd.DataFrame, db_path: str = "data/analyst.db") -> None:
    """Load a DataFrame into SQLite as the 'data' table."""
    global _engine
    _engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql(_table_name, _engine, if_exists="replace", index=False)


def get_engine():
    return _engine


def execute_sql(sql: str) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.
    Raises on error so the caller can handle gracefully.
    """
    if _engine is None:
        raise RuntimeError("No database loaded. Call load_dataframe() first.")
    clean_sql = _clean_sql(sql)
    with _engine.connect() as conn:
        result = pd.read_sql(text(clean_sql), conn)
    return result


def _clean_sql(sql: str) -> str:
    """
    Strip markdown fences, leading/trailing whitespace,
    and limit to SELECT statements only (safety).
    """
    sql = sql.strip()
    # Remove ```sql ... ``` or ``` ... ``` markdown fences
    sql = re.sub(r"```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = sql.rstrip("`").strip()

    # Only allow SELECT for safety
    if not sql.upper().lstrip().startswith("SELECT"):
        raise ValueError(
            "Only SELECT statements are allowed for safety. "
            f"Got: {sql[:60]}..."
        )
    return sql


def build_nl_to_sql_prompt(question: str, columns: list, sample_rows: list) -> str:
    """
    Build the system + user prompt for NL → SQL conversion.
    """
    cols_str = ", ".join(columns)
    sample_str = "\n".join(str(r) for r in sample_rows[:3])

    return f"""You are an expert SQL assistant. Convert the user's question into a valid SQLite SELECT query.

Table name: {_table_name}
Columns: {cols_str}

Sample rows:
{sample_str}

Rules:
- Return ONLY the raw SQL query, no explanation, no markdown fences.
- Use SQLite syntax.
- Only use SELECT statements.
- Wrap column names with spaces in double quotes.
- If aggregation is needed (sum, count, avg), include appropriate GROUP BY.
- Limit results to 500 rows maximum using LIMIT.

Question: {question}
SQL:"""
