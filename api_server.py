"""
FastAPI backend for the React data analysis dashboard.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="AI Data Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROW_LIMIT = 5000
DATASETS: dict[str, pd.DataFrame] = {}
DATASET_NAMES: dict[str, str] = {}
ROOT = Path(__file__).resolve().parent


class DatasetRequest(BaseModel):
    dataset_id: str


class InsightRequest(DatasetRequest):
    focus_column: str | None = None


class BasicPredictionRequest(DatasetRequest):
    target_column: str
    feature_columns: list[str]


class AdvancedPredictionRequest(DatasetRequest):
    target_column: str


def read_tabular_file(filename: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buffer = io.BytesIO(payload)

    if suffix == ".csv":
        return pd.read_csv(buffer)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(buffer)

    raise HTTPException(status_code=400, detail="Upload a CSV, XLS, or XLSX file.")


def get_dataset(dataset_id: str) -> pd.DataFrame:
    df = DATASETS.get(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload it again.")
    return df


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if np.isnan(number) or np.isinf(number):
        return None
    return number


def records_for_frontend(df: pd.DataFrame, limit: int = DATA_ROW_LIMIT) -> list[dict[str, Any]]:
    cleaned = df.head(limit).replace([np.inf, -np.inf], np.nan)
    return json.loads(cleaned.to_json(orient="records", date_format="iso"))


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["number"]).columns)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["object", "category", "bool"]).columns)


def build_column_info(df: pd.DataFrame) -> list[dict[str, Any]]:
    missing = df.isnull().sum()
    return [
        {
            "column": column,
            "dtype": str(df[column].dtype),
            "missing": int(missing[column]),
            "unique": int(df[column].nunique(dropna=True)),
        }
        for column in df.columns
    ]


def build_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in numeric_columns(df):
        series = df[column]
        rows.append(
            {
                "column": column,
                "count": int(series.count()),
                "mean": safe_float(series.mean()),
                "median": safe_float(series.median()),
                "std": safe_float(series.std()),
                "min": safe_float(series.min()),
                "max": safe_float(series.max()),
                "sum": safe_float(series.sum()),
            }
        )
    return rows


def build_quality(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isnull().sum()
    return {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "missing": {column: int(missing[column]) for column in df.columns},
        "totalMissing": int(missing.sum()),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }


def build_correlation(df: pd.DataFrame) -> dict[str, Any] | None:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        return None

    corr_matrix = numeric_df.corr()
    labels = list(corr_matrix.columns)
    matrix = [
        [safe_float(corr_matrix.iloc[row_idx, col_idx]) for col_idx in range(len(labels))]
        for row_idx in range(len(labels))
    ]

    strong: list[dict[str, Any]] = []
    for row_idx in range(len(labels)):
        for col_idx in range(row_idx + 1, len(labels)):
            correlation = safe_float(corr_matrix.iloc[row_idx, col_idx])
            if correlation is not None and abs(correlation) > 0.7:
                strong.append(
                    {
                        "col1": labels[row_idx],
                        "col2": labels[col_idx],
                        "correlation": correlation,
                    }
                )

    return {
        "labels": labels,
        "matrix": matrix,
        "strongCorrelations": sorted(
            strong, key=lambda item: abs(item["correlation"]), reverse=True
        ),
    }


def build_dataset_payload(dataset_id: str, name: str, df: pd.DataFrame) -> dict[str, Any]:
    numeric = numeric_columns(df)
    categorical = categorical_columns(df)
    records = records_for_frontend(df)

    return {
        "datasetId": dataset_id,
        "name": name,
        "columns": list(df.columns),
        "numericColumns": numeric,
        "categoricalColumns": categorical,
        "overview": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "numericColumns": len(numeric),
            "categoricalColumns": len(categorical),
            "missingValues": int(df.isnull().sum().sum()),
            "recordLimit": DATA_ROW_LIMIT,
            "isLimited": df.shape[0] > DATA_ROW_LIMIT,
        },
        "columnInfo": build_column_info(df),
        "stats": build_stats(df),
        "quality": build_quality(df),
        "correlation": build_correlation(df),
        "records": records,
    }


def local_insights(df: pd.DataFrame) -> str:
    parts: list[str] = []
    overview = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns."
    parts.append(overview)

    missing_total = int(df.isnull().sum().sum())
    if missing_total:
        parts.append(f"There are {missing_total} missing values to review before modeling.")
    else:
        parts.append("No missing values were detected.")

    stats = build_stats(df)
    if stats:
        highest_sum = max(stats, key=lambda row: row["sum"] or 0)
        parts.append(
            f"{highest_sum['column']} has the largest numeric total at "
            f"{highest_sum['sum']:.2f}."
        )

    correlation = build_correlation(df)
    if correlation and correlation["strongCorrelations"]:
        strongest = correlation["strongCorrelations"][0]
        parts.append(
            f"The strongest relationship is {strongest['col1']} vs "
            f"{strongest['col2']} with r={strongest['correlation']:.3f}."
        )

    categorical = categorical_columns(df)
    numeric = numeric_columns(df)
    if categorical and numeric:
        category = categorical[0]
        value = numeric[0]
        grouped = df.groupby(category, dropna=False)[value].sum().sort_values(ascending=False)
        if not grouped.empty:
            leader = grouped.index[0]
            parts.append(f"{leader} leads by total {value} when grouped by {category}.")

    return "\n\n".join(parts)


def generate_ai_insights(df: pd.DataFrame, focus_column: str | None = None) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return local_insights(df)

    try:
        from langchain_groq import ChatGroq

        columns = list(df.columns)
        sample = df.head(10).to_string()
        stats = build_stats(df)
        focus_text = f"Focus on {focus_column}." if focus_column else ""

        prompt = f"""Analyze this dataset and provide concise business insights.

Columns: {columns}
Sample data:
{sample}

Statistics:
{json.dumps(stats, indent=2)}

{focus_text}

Return:
1. Key patterns
2. Notable outliers or quality concerns
3. Recommended next actions
"""

        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)
        return llm.invoke(prompt).content
    except Exception as exc:
        return f"{local_insights(df)}\n\nAI provider error: {exc}"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    payload = await file.read()
    df = read_tabular_file(file.filename or "dataset.csv", payload)
    dataset_id = str(uuid.uuid4())

    DATASETS[dataset_id] = df
    DATASET_NAMES[dataset_id] = file.filename or "Uploaded dataset"

    return build_dataset_payload(dataset_id, DATASET_NAMES[dataset_id], df)


@app.post("/api/sample")
def load_sample_dataset() -> dict[str, Any]:
    sample_path = ROOT / "sample_data.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="sample_data.csv was not found.")

    df = pd.read_csv(sample_path)
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = df
    DATASET_NAMES[dataset_id] = sample_path.name
    return build_dataset_payload(dataset_id, sample_path.name, df)


@app.get("/api/datasets/{dataset_id}")
def dataset(dataset_id: str) -> dict[str, Any]:
    df = get_dataset(dataset_id)
    return build_dataset_payload(dataset_id, DATASET_NAMES.get(dataset_id, "Dataset"), df)


@app.post("/api/insights")
def insights(request: InsightRequest) -> dict[str, str]:
    df = get_dataset(request.dataset_id)
    return {"insights": generate_ai_insights(df, request.focus_column)}


@app.post("/api/predict/basic")
def basic_prediction(request: BasicPredictionRequest) -> dict[str, Any]:
    from sklearn.linear_model import LinearRegression

    df = get_dataset(request.dataset_id)
    df_numeric = df.select_dtypes(include=["number"]).copy()

    if request.target_column not in df_numeric.columns:
        raise HTTPException(status_code=400, detail="Target column must be numeric.")

    features = [
        column
        for column in request.feature_columns
        if column in df_numeric.columns and column != request.target_column
    ]
    if not features:
        raise HTTPException(status_code=400, detail="Choose at least one numeric feature column.")

    clean = df_numeric[[request.target_column, *features]].dropna()
    if clean.shape[0] < 2:
        raise HTTPException(status_code=400, detail="Not enough complete rows for prediction.")

    x = clean[features]
    y = clean[request.target_column]
    model = LinearRegression()
    model.fit(x, y)

    return {
        "target": request.target_column,
        "features": features,
        "score": safe_float(model.score(x, y)),
        "intercept": safe_float(model.intercept_),
        "coefficients": {
            feature: safe_float(coef) for feature, coef in zip(features, model.coef_)
        },
    }


@app.post("/api/predict/advanced")
def advanced_prediction(request: AdvancedPredictionRequest) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    df = get_dataset(request.dataset_id)
    df_numeric = df.select_dtypes(include=["number"]).copy().dropna()

    if request.target_column not in df_numeric.columns:
        raise HTTPException(status_code=400, detail="Target column must be numeric.")

    feature_columns = [column for column in df_numeric.columns if column != request.target_column]
    if not feature_columns:
        raise HTTPException(status_code=400, detail="No numeric feature columns found.")
    if df_numeric.shape[0] < 5:
        raise HTTPException(status_code=400, detail="Need at least five complete rows.")

    x = df_numeric[feature_columns]
    y = df_numeric[request.target_column]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results: list[dict[str, Any]] = []
    for name, model in models.items():
        if name in {"Ridge Regression", "Lasso Regression"}:
            model.fit(x_train_scaled, y_train)
            predictions = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

        mse = mean_squared_error(y_test, predictions)
        results.append(
            {
                "name": name,
                "r2": safe_float(r2_score(y_test, predictions)),
                "rmse": safe_float(np.sqrt(mse)),
                "mae": safe_float(mean_absolute_error(y_test, predictions)),
            }
        )

    results.sort(key=lambda item: item["r2"] if item["r2"] is not None else -999, reverse=True)
    best = results[0]

    return {
        "target": request.target_column,
        "features": feature_columns,
        "bestModel": best["name"],
        "bestR2": best["r2"],
        "results": results,
    }
