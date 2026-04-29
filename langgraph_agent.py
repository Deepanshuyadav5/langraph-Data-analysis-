"""
LangGraph Data Analysis Agent
Uses Groq LLM for intelligent data analysis
"""

import os
import pandas as pd
import numpy as np
from typing import TypedDict, Annotated, Sequence, List
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Read Groq API key from the environment. Set GROQ_API_KEY before running
# LangGraph-powered analysis.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY) if GROQ_API_KEY else None


class AgentState(TypedDict):
    """State for the data analysis agent"""
    messages: Sequence[BaseMessage]
    dataframe: pd.DataFrame | None
    analysis_results: dict | None
    visualizations: list | None
    current_file: str | None
    predictions: dict | None
    model_accuracy: float | None


def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file into DataFrame"""
    return pd.read_csv(file_path)


def get_dataframe_info(df: pd.DataFrame) -> str:
    """Get information about the dataframe"""
    info = f"Shape: {df.shape}\n"
    info += f"Columns: {list(df.columns)}\n"
    info += f"Data Types:\n{df.dtypes.to_string()}\n"
    info += f"\nFirst 5 rows:\n{df.head().to_string()}\n"
    return info


def calculate_basic_stats(df: pd.DataFrame) -> dict:
    """Calculate basic statistics"""
    stats = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        stats[col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'sum': float(df[col].sum())
        }
    
    return stats


def analyze_by_category(df: pd.DataFrame, category_col: str, value_col: str) -> dict:
    """Analyze data by category"""
    if category_col not in df.columns or value_col not in df.columns:
        return {"error": "Invalid columns"}
    
    result = df.groupby(category_col)[value_col].agg(['sum', 'mean', 'count']).to_dict()
    return result


def create_visualization(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str, 
                        title: str = "", output_dir: str = "charts") -> str:
    """Create and save visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    if chart_type == "bar":
        df.groupby(x_col)[y_col].sum().plot(kind='bar', color='skyblue')
    elif chart_type == "line":
        df.plot(x=x_col, y=y_col, kind='line', marker='o', color='green')
    elif chart_type == "scatter":
        plt.scatter(df[x_col], df[y_col], alpha=0.6, color='coral')
    elif chart_type == "histogram":
        df[y_col].hist(bins=20, color='purple', alpha=0.7)
    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include=['number'])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    elif chart_type == "pie":
        df.groupby(x_col)[y_col].sum().plot(kind='pie', autopct='%1.1f%%')
    elif chart_type == "box":
        df.boxplot(column=y_col, by=x_col, figsize=(10, 6))
        plt.title(title or f"{y_col} by {x_col}")
        plt.suptitle('')
    else:
        return "Invalid chart type"
    
    plt.title(title or f"{y_col} by {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    
    filename = f"{output_dir}/{chart_type}_{x_col}_{y_col}.png"
    plt.savefig(filename)
    plt.close()
    
    return filename


@tool
def analyze_data(file_path: str) -> str:
    """Load and analyze a CSV file"""
    try:
        df = load_csv(file_path)
        info = get_dataframe_info(df)
        stats = calculate_basic_stats(df)
        
        result = f"Data loaded successfully!\n\n{info}\n\nStatistics:\n{json.dumps(stats, indent=2)}"
        return result
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_columns(file_path: str) -> str:
    """Get column names from a CSV file"""
    try:
        df = load_csv(file_path)
        return f"Columns: {list(df.columns)}\nNumeric: {list(df.select_dtypes(include=['number']).columns)}\nCategorical: {list(df.select_dtypes(include=['object']).columns)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def generate_insights(file_path: str, focus_column: str = None) -> str:
    """Generate AI-powered insights from the data"""
    try:
        if llm is None:
            return "Error: Set the GROQ_API_KEY environment variable before generating AI insights."

        df = load_csv(file_path)
        
        # Get basic stats
        stats = calculate_basic_stats(df)
        
        # Build prompt for LLM
        columns = list(df.columns)
        sample = df.head(10).to_string()
        
        prompt = f"""Analyze this dataset and provide insights:
        Columns: {columns}
        Sample data:
        {sample}
        
        Key statistics:
        {json.dumps(stats, indent=2)}
        
        Provide:
        1. Key patterns and trends
        2. Top performing categories
        3. Areas for improvement
        4. Actionable recommendations
        """
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def create_chart(file_path: str, chart_type: str, x_column: str, 
                 y_column: str, title: str = "") -> str:
    """Create a visualization chart"""
    try:
        df = load_csv(file_path)
        filename = create_visualization(df, chart_type, x_column, y_column, title)
        return f"Chart saved to: {filename}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def compare_columns(file_path: str, col1: str, col2: str) -> str:
    """Compare two columns and provide analysis"""
    try:
        df = load_csv(file_path)
        
        if col1 not in df.columns or col2 not in df.columns:
            return "Invalid column names"
        
        # Numeric comparison
        if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
            correlation = df[col1].corr(df[col2])
            return f"Correlation between {col1} and {col2}: {correlation:.4f}"
        
        # Categorical comparison
        cross_tab = pd.crosstab(df[col1], df[col2])
        return f"Cross-tabulation:\n{cross_tab.to_string()}"
        
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def predict_value(file_path: str, target_column: str, features: list) -> str:
    """Simple prediction based on historical data patterns"""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        df = load_csv(file_path)
        
        if target_column not in df.columns:
            return f"Column {target_column} not found"
        
        # Prepare data
        df_numeric = df.select_dtypes(include=['number']).copy()
        
        if df_numeric.empty or target_column not in df_numeric.columns:
            return "Need numeric columns for prediction"
        
        # Simple prediction using linear regression
        feature_cols = [f for f in features if f in df_numeric.columns and f != target_column]
        
        if not feature_cols:
            return "No valid feature columns found"
        
        X = df_numeric[feature_cols]
        y = df_numeric[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(feature_cols, np.abs(model.coef_)))
        
        return f"""Prediction model created!
        
Target: {target_column}
Features: {feature_cols}

Model Performance:
- R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)
- RMSE: {rmse:.4f}
- MAE: {mae:.4f}

Feature Importance:
{json.dumps(feature_importance, indent=2)}

Coefficients:
{json.dumps(dict(zip(feature_cols, model.coef_)), indent=2)}

This model explains {r2*100:.2f}% of the variance in {target_column}"""
        
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def advanced_prediction(file_path: str, target_column: str) -> str:
    """Advanced prediction with multiple models and cross-validation"""
    try:
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        from sklearn.preprocessing import StandardScaler
        
        df = load_csv(file_path)
        
        if target_column not in df.columns:
            return f"Column {target_column} not found"
        
        # Prepare data
        df_numeric = df.select_dtypes(include=['number']).copy()
        
        if df_numeric.empty or target_column not in df_numeric.columns:
            return "Need numeric columns for prediction"
        
        feature_cols = [f for f in df_numeric.columns if f != target_column]
        
        if not feature_cols:
            return "No valid feature columns found"
        
        X = df_numeric[feature_cols]
        y = df_numeric[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        best_model = None
        best_r2 = -float('inf')
        
        for name, model in models.items():
            # Train
            if 'Ridge' in name or 'Lasso' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae
            }
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = name
        
        # Format results
        result = f"Advanced Prediction Analysis for '{target_column}'\n"
        result += "=" * 60 + "\n\n"
        result += "Model Comparison:\n"
        result += "-" * 60 + "\n"
        
        for name, metrics in sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True):
            result += f"\n{name}:\n"
            result += f"  R² Score: {metrics['R2']:.4f} ({metrics['R2']*100:.2f}%)\n"
            result += f"  RMSE: {metrics['RMSE']:.4f}\n"
            result += f"  MAE: {metrics['MAE']:.4f}\n"
        
        result += f"\n{'=' * 60}\n"
        result += f"Best Model: {best_model}\n"
        result += f"Best R² Score: {best_r2:.4f} ({best_r2*100:.2f}%)\n"
        result += f"{'=' * 60}\n"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def time_series_forecast(file_path: str, target_column: str, periods: int = 12) -> str:
    """Time series forecasting using simple moving average and trend analysis"""
    try:
        df = load_csv(file_path)
        
        if target_column not in df.columns:
            return f"Column {target_column} not found"
        
        # Check if numeric
        if df[target_column].dtype not in ['int64', 'float64']:
            return f"Column {target_column} is not numeric"
        
        values = df[target_column].values
        
        # Simple Moving Average
        window = min(3, len(values))
        sma = pd.Series(values).rolling(window=window).mean().values
        
        # Calculate trend
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Forecast
        future_x = np.arange(len(values), len(values) + periods)
        forecast = slope * future_x + intercept
        
        # Calculate accuracy metrics
        sma_aligned = sma[~np.isnan(sma)]
        actual_aligned = values[len(sma)-len(sma_aligned):]
        
        if len(sma_aligned) > 0:
            mae = np.mean(np.abs(actual_aligned - sma_aligned))
            mape = np.mean(np.abs((actual_aligned - sma_aligned) / actual_aligned)) * 100
        else:
            mae = 0
            mape = 0
        
        result = f"Time Series Forecast for '{target_column}'\n"
        result += "=" * 50 + "\n\n"
        result += f"Historical Statistics:\n"
        result += f"  Mean: {np.mean(values):.2f}\n"
        result += f"  Std Dev: {np.std(values):.2f}\n"
        result += f"  Min: {np.min(values):.2f}\n"
        result += f"  Max: {np.max(values):.2f}\n\n"
        
        result += f"Trend Analysis:\n"
        result += f"  Slope: {slope:.4f}\n"
        result += f"  Direction: {'Upward' if slope > 0 else 'Downward'}\n\n"
        
        result += f"Moving Average (window={window}):\n"
        result += f"  MAE: {mae:.2f}\n"
        result += f"  MAPE: {mape:.2f}%\n\n"
        
        result += f"Forecast for next {periods} periods:\n"
        result += "-" * 50 + "\n"
        for i, val in enumerate(forecast, 1):
            result += f"  Period {i}: {val:.2f}\n"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def data_quality_check(file_path: str) -> str:
    """Check data quality and provide report"""
    try:
        df = load_csv(file_path)
        
        result = f"Data Quality Report for '{file_path}'\n"
        result += "=" * 50 + "\n\n"
        
        # Shape
        result += f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        result += "Missing Values:\n"
        for col in df.columns:
            if missing[col] > 0:
                result += f"  {col}: {missing[col]} ({missing_pct[col]:.2f}%)\n"
        if missing.sum() == 0:
            result += "  No missing values found!\n"
        result += "\n"
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        result += f"Duplicate Rows: {duplicates}\n\n"
        
        # Data types
        result += "Data Types:\n"
        for col in df.columns:
            result += f"  {col}: {df[col].dtype}\n"
        result += "\n"
        
        # Numeric columns stats
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            result += "Numeric Columns Summary:\n"
            result += df[numeric_cols].describe().to_string() + "\n\n"
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            result += "Categorical Columns:\n"
            for col in cat_cols:
                unique = df[col].nunique()
                result += f"  {col}: {unique} unique values\n"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def correlation_analysis(file_path: str) -> str:
    """Perform correlation analysis on numeric columns"""
    try:
        df = load_csv(file_path)
        
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return "Need at least 2 numeric columns for correlation analysis"
        
        corr_matrix = numeric_df.corr()
        
        result = f"Correlation Analysis\n"
        result += "=" * 50 + "\n\n"
        result += "Correlation Matrix:\n"
        result += corr_matrix.to_string() + "\n\n"
        
        # Find strong correlations
        result += "Strong Correlations (|r| > 0.7):\n"
        result += "-" * 50 + "\n"
        
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if strong_corrs:
            for col1, col2, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
                result += f"  {col1} ↔ {col2}: {corr:.4f}\n"
        else:
            result += "  No strong correlations found\n"
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"


# Tool list for the agent
tools = [
    analyze_data, 
    get_columns, 
    generate_insights, 
    create_chart, 
    compare_columns, 
    predict_value,
    advanced_prediction,
    time_series_forecast,
    data_quality_check,
    correlation_analysis
]
tool_node = ToolNode(tools)


def should_continue(state: AgentState) -> bool:
    """Determine if the agent should continue"""
    return len(state["messages"]) > 3


def analyze_node(state: AgentState) -> AgentState:
    """Node for analyzing data with LLM"""
    if llm is None:
        return {
            "messages": state["messages"] + [
                AIMessage(content="Set the GROQ_API_KEY environment variable before running the agent.")
            ],
            "dataframe": state.get("dataframe"),
            "analysis_results": state.get("analysis_results"),
            "visualizations": state.get("visualizations"),
            "current_file": state.get("current_file")
        }

    messages = state["messages"]
    
    # Get the last user message
    user_message = messages[-1].content if messages else ""
    
    # Call LLM with tools
    ai_message = llm.bind_tools(tools).invoke(messages + [HumanMessage(content=user_message)])
    
    return {
        "messages": state["messages"] + [ai_message],
        "dataframe": state.get("dataframe"),
        "analysis_results": state.get("analysis_results"),
        "visualizations": state.get("visualizations"),
        "current_file": state.get("current_file")
    }


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("analyzer", analyze_node)
graph.add_node("tools", tool_node)

graph.add_edge("__start__", "analyzer")
graph.add_conditional_edges(
    "analyzer",
    should_continue,
    {
        True: "tools",
        False: END
    }
)
graph.add_edge("tools", "analyzer")

agent = graph.compile()


def run_analysis(file_path: str, user_query: str) -> str:
    """Run the data analysis agent"""
    initial_state = {
        "messages": [
            HumanMessage(content=f"Analyze this data file: {file_path}. {user_query}")
        ],
        "dataframe": None,
        "analysis_results": {},
        "visualizations": [],
        "current_file": file_path
    }
    
    result = agent.invoke(initial_state)
    return result["messages"][-1].content


if __name__ == "__main__":
    # Test with sample data
    print("Testing LangGraph Data Analysis Agent...")
    result = run_analysis("sample_data.csv", "Give me a summary of this data")
    print(result)
