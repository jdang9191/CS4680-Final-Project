from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Any

import pandas as pd

app = FastAPI(title="CSV Cleaning MCP Server")


# This file is the deterministic tool layer. The planner can suggest actions,
# but actual CSV mutations happen only through these typed endpoints.
MISSING_TOKENS_EXCEPT_UNKNOWN = {
    "error",
    "n/a",
    "na",
    "null",
    "nan",
    "missing",
}


# --------------------------
# Input Schemas
# --------------------------

#typed schemas make tool calls explicit and auditable.
class FileInput(BaseModel):
    """Simple input for tools that only need a CSV file path."""
    file_path: str
    output_file_path: str | None = None


class ImputeInput(BaseModel):
    """Input for a single-column imputation operation."""
    file_path: str
    column: str
    strategy: str
    fill_value: Any | None = None
    output_file_path: str | None = None


class DropColumnsInput(BaseModel):
    """Input for dropping one or more columns from the CSV."""
    file_path: str
    columns: list[str]
    output_file_path: str | None = None


class DropRowsWithNullsInput(BaseModel):
    """Input for dropping rows with null values, optionally limited to selected columns."""
    file_path: str
    columns: list[str] | None = None
    output_file_path: str | None = None


# --------------------------
# Helpers
# --------------------------
# Data sanitation starts here by normalizing common dirty tokens to nulls.
# This keeps downstream tool behavior consistent.
def _load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file or raise a clear HTTP error."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="File not found.")

    try:
        df = pd.read_csv(file_path)

        # Normalize dirty placeholders to true nulls while preserving "unknown" as a valid category.
        for column in df.columns:
            if not (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])):
                continue

            normalized = df[column].map(
                lambda value: value.strip().lower() if isinstance(value, str) else value
            )
            df[column] = df[column].mask(normalized.isin(MISSING_TOKENS_EXCEPT_UNKNOWN), pd.NA)

        return df
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc


def _build_output_path(file_path: str, output_file_path: str | None, label: str) -> str:
    """Create a new output CSV path so the original file is preserved."""
    if output_file_path:
        return output_file_path

    directory = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    candidate = os.path.join(directory, f"{base_name}_{label}.csv")

    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base_name}_{label}_{counter}.csv")
        counter += 1

    return candidate


def _numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return basic summary stats for numeric columns."""
    summary: dict[str, dict[str, Any]] = {}

    for column in df.select_dtypes(include="number").columns:
        series = df[column].dropna()
        summary[column] = {
            "mean": None if series.empty else float(series.mean()),
            "median": None if series.empty else float(series.median()),
            "std": None if series.empty else float(series.std(ddof=0)),
            "min": None if series.empty else float(series.min()),
            "max": None if series.empty else float(series.max()),
            "skew": None if series.empty else float(series.skew()) if len(series) > 2 else 0.0,
        }

    return summary


def _column_support_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Describe which operations are valid for each column type."""
    summary: dict[str, dict[str, Any]] = {}

    for column in df.columns:
        dtype_name = str(df[column].dtype)
        if pd.api.types.is_numeric_dtype(df[column]):
            allowed_operations = ["mean", "median", "mode", "constant"]
            supported = True
        elif pd.api.types.is_bool_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            allowed_operations = ["mode", "constant"]
            supported = True
        else:
            allowed_operations = []
            supported = False

        summary[column] = {
            "dtype": dtype_name,
            "supported": supported,
            "allowed_operations": allowed_operations,
        }

    return summary


# --------------------------
# Tool 1: profile_dataset
# --------------------------

# Profiling is the planner's evidence source: row counts, missingness,
# duplicates, column types, and numeric statistics.
@app.post("/profile_dataset")
def profile_dataset(data: FileInput):
    """Inspect the CSV so the cleaner agent can reason about it."""

    df = _load_csv(data.file_path)

    categorical_columns = list(df.select_dtypes(exclude="number").columns)
    numeric_columns = list(df.select_dtypes(include="number").columns)

    return {
        # Basic structure.
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        # Column-by-column support metadata.
        "column_support": _column_support_summary(df),
        # Missingness and duplicate signals.
        "missing_percentage": df.isnull().mean().to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "duplicate_percentage": float(df.duplicated().mean()) if len(df) else 0.0,
        # Extra stats the cleaner can use for imputation choices.
        "numeric_summary": _numeric_summary(df),
        # Useful for deciding whether a categorical column is too sparse/high-cardinality.
        "cardinality": {column: int(df[column].nunique(dropna=True)) for column in categorical_columns},
    }


# --------------------------
# Tool 2: remove_duplicates
# --------------------------

#Deterministic operation with measurable before/after.
@app.post("/remove_duplicates")
def remove_duplicates(data: FileInput):
    """Remove exact duplicate rows and write the result to a new CSV."""
    df = _load_csv(data.file_path)

    original_rows = len(df)
    df_clean = df.drop_duplicates()
    rows_removed = original_rows - len(df_clean)

    output_file_path = _build_output_path(data.file_path, data.output_file_path, "cleaned_deduped")

    df_clean.to_csv(output_file_path, index=False)

    return {
        "status": "success",
        "rows_removed": rows_removed,
        "rows_before": original_rows,
        "rows_after": len(df_clean),
        "output_file_path": output_file_path,
    }


# --------------------------
# Tool 3: impute_missing_values
# --------------------------

# Imputation is constrained by strategy + column type checks. This prevents
# invalid operations like mean on text columns.
@app.post("/impute_missing_values")
def impute_missing_values(data: ImputeInput):
    """Fill missing values in one column and write the result to a new CSV."""

    df = _load_csv(data.file_path)

    if data.column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found.")

    if data.strategy not in ["mean", "median", "mode", "constant"]:
        raise HTTPException(status_code=400, detail="Invalid strategy.")

    series = df[data.column]
    missing_before = int(series.isnull().sum())

    def _maybe_round_whole_number(value: Any) -> Any:
        """Round numeric imputation results to the nearest whole number."""
        if isinstance(value, (int, float)):
            return int(round(float(value)))
        return value

    # Mean and median only make sense for numeric columns.
    if data.strategy in ["mean", "median"] and not pd.api.types.is_numeric_dtype(series):
        raise HTTPException(status_code=400, detail=f"{data.strategy.title()} imputation requires a numeric column.")

    if data.strategy == "mean":
        imputed_value = _maybe_round_whole_number(series.mean())
        df[data.column] = series.fillna(imputed_value)
    elif data.strategy == "median":
        imputed_value = _maybe_round_whole_number(series.median())
        df[data.column] = series.fillna(imputed_value)
    elif data.strategy == "constant":
        if data.fill_value is None:
            raise HTTPException(status_code=400, detail="Constant imputation requires fill_value.")
        imputed_value = data.fill_value
        df[data.column] = series.fillna(imputed_value)
    else:
        # Mode works for text/categorical columns and also for numeric columns if needed.
        modes = series.mode(dropna=True)
        if modes.empty:
            raise HTTPException(status_code=400, detail="Cannot impute mode for an empty column.")
        imputed_value = _maybe_round_whole_number(modes.iloc[0])
        df[data.column] = series.fillna(imputed_value)

    # Keep numeric columns with whole-number imputations looking like whole numbers in CSV output.
    if pd.api.types.is_numeric_dtype(df[data.column]):
        non_null_values = df[data.column].dropna()
        if not non_null_values.empty and (non_null_values % 1 == 0).all():
            df[data.column] = df[data.column].round().astype("Int64")

    output_file_path = _build_output_path(data.file_path, data.output_file_path, f"cleaned_imputed_{data.column}")

    df.to_csv(output_file_path, index=False)

    return {
        "status": "success",
        "column": data.column,
        "strategy": data.strategy,
        "imputed_value": imputed_value,
        "missing_before": missing_before,
        "missing_after": int(df[data.column].isnull().sum()),
        "output_file_path": output_file_path,
    }


# --------------------------
# Tool 4: drop_columns
# --------------------------

#Schema validation + existence checks before mutation.
@app.post("/drop_columns")
def drop_columns(data: DropColumnsInput):
    """Remove entire columns and write the result to a new CSV."""
    df = _load_csv(data.file_path)

    missing_columns = [column for column in data.columns if column not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Column not found: {missing_columns}")

    original_columns = list(df.columns)
    df_clean = df.drop(columns=data.columns)
    output_file_path = _build_output_path(data.file_path, data.output_file_path, "cleaned_dropped_columns")
    df_clean.to_csv(output_file_path, index=False)

    return {
        "status": "success",
        "dropped_columns": data.columns,
        "columns_before": original_columns,
        "columns_after": list(df_clean.columns),
        "output_file_path": output_file_path,
    }


# --------------------------
# Tool 5: drop_rows_with_nulls
# --------------------------

# This endpoint executes row deletion deterministically; in the orchestrator,
# prompt-level guardrails decide whether broad row dropping is allowed.
@app.post("/drop_rows_with_nulls")
def drop_rows_with_nulls(data: DropRowsWithNullsInput):
    """Drop rows that contain null values, either in any column or in selected columns."""

    df = _load_csv(data.file_path)
    original_rows = len(df)

    subset_columns = data.columns
    if subset_columns:
        missing_columns = [column for column in subset_columns if column not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Column not found: {missing_columns}")
        df_clean = df.dropna(subset=subset_columns)
    else:
        df_clean = df.dropna()

    rows_removed = original_rows - len(df_clean)
    output_file_path = _build_output_path(data.file_path, data.output_file_path, "cleaned_drop_null_rows")
    df_clean.to_csv(output_file_path, index=False)

    return {
        "status": "success",
        "mode": "selected_columns" if subset_columns else "any_column",
        "columns": subset_columns or [],
        "rows_before": original_rows,
        "rows_after": len(df_clean),
        "rows_removed": rows_removed,
        "output_file_path": output_file_path,
    }
