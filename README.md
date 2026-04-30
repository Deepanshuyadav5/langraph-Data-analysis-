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

## Deploy On Render

This project should be deployed as **two Render services**:

1. A FastAPI backend Web Service
2. A React frontend Static Site

The frontend calls the backend through the `VITE_API_URL` environment variable.

### 1. Push Code To GitHub

Make sure the latest code is pushed to GitHub:

```powershell
git status
git add .
git commit -m "Update Render deployment docs"
git push
```

### 2. Deploy Backend Web Service

In Render:

1. Open the Render dashboard.
2. Click **New +**.
3. Select **Web Service**.
4. Connect your GitHub repository.
5. Select the repo branch, usually `main`.
6. Use these settings:

```text
Name: langgraph-data-analysis-api
Runtime: Python 3
Root Directory: leave empty
Build Command: python -m pip install -r requirements.txt
Start Command: python -m uvicorn api_server:app --host 0.0.0.0 --port $PORT
```

Add these backend environment variables:

```text
GROQ_API_KEY=your_groq_api_key_here
CORS_ORIGINS=https://your-frontend-service-name.onrender.com
```

`GROQ_API_KEY` is optional. Without it, the backend still runs and returns local fallback insights.

After deployment, Render gives you a backend URL like:

```text
https://langgraph-data-analysis-api.onrender.com
```

Test the backend:

```text
https://langgraph-data-analysis-api.onrender.com/api/health
```

Expected response:

```json
{"status":"ok"}
```

### 3. Deploy Frontend Static Site

In Render:

1. Click **New +**.
2. Select **Static Site**.
3. Connect the same GitHub repository.
4. Use these settings:

```text
Name: langgraph-data-analysis-dashboard
Root Directory: frontend
Build Command: npm install && npm run build
Publish Directory: dist
```

Add this frontend environment variable:

```text
VITE_API_URL=https://your-backend-service-name.onrender.com
```

Replace the value with your real backend URL from Render.

After deployment, Render gives you a frontend URL like:

```text
https://langgraph-data-analysis-dashboard.onrender.com
```

### 4. Update Backend CORS

After the frontend URL is created, go back to the backend Web Service and update:

```text
CORS_ORIGINS=https://your-frontend-service-name.onrender.com
```

If you want to allow both local development and Render frontend, use comma-separated values:

```text
CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173,https://your-frontend-service-name.onrender.com
```

Redeploy the backend after changing this variable.

### 5. Final Render URLs

Your final app has two URLs:

```text
Frontend: https://your-frontend-service-name.onrender.com
Backend:  https://your-backend-service-name.onrender.com
```

Users should open the frontend URL.

### Render Troubleshooting

If the frontend opens but uploads fail:

- Check that `VITE_API_URL` points to the backend URL.
- Check that backend `CORS_ORIGINS` includes the frontend URL.
- Open `/api/health` on the backend URL.

If backend deploy fails:

- Confirm `requirements.txt` exists in the repo root.
- Confirm the start command uses `$PORT`.
- Confirm the service runtime is Python 3.

If frontend deploy fails:

- Confirm the static site root directory is `frontend`.
- Confirm the build command is `npm install && npm run build`.
- Confirm the publish directory is `dist`.

## Notes

- Use `python -m pip install -r requirements.txt`, not `pip install requirements.txt`.
- Use `npm run dev`, not `npm start`.
- The frontend expects the backend at `http://127.0.0.1:8000`.
- Uploaded datasets are stored in memory during the backend session. Restarting the backend clears uploaded datasets.
