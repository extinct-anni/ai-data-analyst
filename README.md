# 🧠 AI-Powered Autonomous Data Analyst

Upload any CSV/Excel → auto-clean → ask questions in plain English → get SQL + charts + insights.

## 🚀 Quick Start

```bash
git clone <your-repo>
cd ai-data-analyst

pip install -r requirements.txt

streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Configuration

### Option A — Enter API key in the sidebar (easiest)
Just paste your OpenAI key in the sidebar when the app loads.

### Option B — Environment variable
```bash
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```

### Option C — Streamlit secrets (for cloud deployment)
Edit `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY = "sk-..."
```

---

## 📁 Project Structure

```
ai-data-analyst/
├── app.py                  # Main Streamlit app (all pages)
├── utils/
│   ├── __init__.py
│   ├── schema.py           # Auto-detect column types + stats
│   ├── cleaning.py         # Missing values, outliers, duplicates
│   ├── sql_engine.py       # SQLite loader + query executor
│   ├── ai_engine.py        # All OpenAI API calls
│   └── viz.py              # Chart generation (matplotlib + seaborn)
├── data/                   # SQLite DB stored here at runtime
├── .streamlit/
│   └── secrets.toml        # API key for cloud deployment
├── requirements.txt
└── README.md
```

---

## 🧩 Features by Phase

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | File Upload (CSV/Excel) | ✅ |
| 1 | Auto Schema Detection | ✅ |
| 2 | Missing Value Handling | ✅ |
| 2 | Outlier Detection (IQR + Z-score) | ✅ |
| 2 | Duplicate Removal | ✅ |
| 2 | Auto Data Type Fixing | ✅ |
| 3 | AI Insight Generator | ✅ |
| 3 | AI Dataset Summary | ✅ |
| 4 | Auto Charts from AI | ✅ |
| 4 | Manual Chart Builder | ✅ |
| 5 | NL → SQL Conversion | ✅ |
| 5 | SQL Query Executor | ✅ |
| 6 | Business Result Explanation | ✅ |
| 6 | AI Cleaning Suggestions | ✅ |
| 7 | Polished Dark UI | ✅ |

---

## ☁️ Deployment

### Streamlit Cloud (free, easiest)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as entrypoint
4. Add `OPENAI_API_KEY` in Secrets
5. Deploy ✅

### Render
1. Create a new Web Service
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
4. Add `OPENAI_API_KEY` as environment variable

---

## 💡 Example Questions to Ask

- *"What are the top 5 categories by revenue?"*
- *"Show me sales trends over time"*
- *"Are there any anomalies in the data?"*
- *"Which numeric columns are most correlated?"*
- *"What's the average value grouped by region?"*

---

## 🛠 Tech Stack

- **Streamlit** — UI framework
- **Pandas / NumPy** — Data processing
- **Matplotlib / Seaborn** — Visualizations
- **SQLAlchemy + SQLite** — SQL engine
- **OpenAI GPT-4o-mini** — AI brain
