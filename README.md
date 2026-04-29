# LangGraph Data Analysis Dashboard

An AI-powered data analysis dashboard built with **FastAPI**, **React**, **Plotly**, and **LangGraph/LangChain**. Upload CSV or Excel files, explore dataset quality, generate charts, merge multiple visuals into dashboard-style views, export graphs, and run simple prediction workflows.

## Features

- Upload CSV, XLS, or XLSX datasets
- Load the included sample dataset
- View dataset overview, column metadata, missing values, and numeric statistics
- Generate AI insights with Groq when `GROQ_API_KEY` is available
- Build bar, line, area, scatter, histogram, pie, box, and heatmap charts
- Download individual graphs as PNG
- Build a visualization dashboard
- Merge multiple graphs into one combined visual, similar to a PowerBI-style dashboard
- Export the dashboard as PNG, HTML, or JSON
- Run basic linear regression prediction
- Compare multiple ML models including Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, and Gradient Boosting
- Inspect data quality and strong correlations

## Project Structure

```text
.
├── api_server.py          # FastAPI backend
├── langgraph_agent.py     # LangGraph/LangChain analysis agent
├── requirements.txt       # Python dependencies
├── sample_data.csv        # Sample dataset
├── frontend/              # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.js
└── README.md
```

`streamlit_app.py` is intentionally ignored and not part of the pushed React/FastAPI project.

## Requirements

- Python 3.10+
- Node.js 18+
- npm
- Groq API key for AI insights, optional

## Backend Setup

From the project root:

```powershell
python -m pip install -r requirements.txt
```

Start the backend:

```powershell
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Frontend Setup

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:5173/
```

## Environment Variables

Set `GROQ_API_KEY` to enable live AI insights.

PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

Then start the backend in the same terminal:

```powershell
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

If `GROQ_API_KEY` is not set, the backend still works and returns local rule-based insights.

## Common Commands

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run backend:

```powershell
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Run frontend:

```powershell
npm run dev
```

Build frontend:

```powershell
npm run build
```

## API Endpoints

- `GET /api/health` - backend health check
- `POST /api/upload` - upload CSV or Excel dataset
- `POST /api/sample` - load `sample_data.csv`
- `GET /api/datasets/{dataset_id}` - get dataset profile and records
- `POST /api/insights` - generate AI or fallback insights
- `POST /api/predict/basic` - train basic linear regression model
- `POST /api/predict/advanced` - compare multiple regression models

## Notes

- Use `python -m pip install -r requirements.txt`, not `pip install requirements.txt`.
- Use `npm run dev`, not `npm start`.
- The frontend expects the backend at `http://127.0.0.1:8000`.
- Uploaded datasets are stored in memory during the backend session. Restarting the backend clears uploaded datasets.

