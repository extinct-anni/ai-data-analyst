# 🧠 AI-Powered Autonomous Data Analyst

Upload any CSV/Excel → auto-clean → ask questions in plain English → get SQL + charts + insights.

## 🌐 Live Demo

🔗 **[Launch App](https://ai-data-analyst-fmhssbhdbkyubmkpgozcf7.streamlit.app/)**

---

## 🚀 Quick Start

```bash
git clone https://github.com/extinct-anni/ai-data-analyst
cd ai-data-analyst
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ Configuration

### Option A — Streamlit secrets (recommended)
Edit `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

### Option B — Environment variable
```bash
export GROQ_API_KEY="gsk_your_key_here"
streamlit run app.py
```

Get a free API key at [console.groq.com](https://console.groq.com)

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
│   ├── ai_engine.py        # All Groq AI API calls
│   └── viz.py              # Chart generation
├── data/                   # SQLite DB stored here at runtime
├── .streamlit/
│   └── secrets.toml        # API key for cloud deployment
├── requirements.txt
└── README.md
```

---

## 🧩 Features

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
- **Groq (Llama 3.3-70B)** — Free AI brain

---

## ☁️ Deployment

### Streamlit Cloud (free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entrypoint
4. Add `GROQ_API_KEY` in Secrets
5. Deploy ✅
