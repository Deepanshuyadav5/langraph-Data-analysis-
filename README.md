# LangGraph Data Analysis Dashboard

React + FastAPI dashboard for uploading tabular data, generating insights, creating visualizations, merging graphs, and exporting dashboards.

## Backend

```powershell
python -m pip install -r requirements.txt
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173/`.

Set `GROQ_API_KEY` in your environment to enable live AI insights.
