# Tool Design Document

## Domain: CSV Data Cleaning Assistant

### Overview

This MCP Server implements tools for a CSV Data Cleaning Assistant. The server exposes structured tools that allow an AI host to profile datasets, remove duplicate rows, and impute missing values using clearly defined input and output schemas.

The purpose of this server is to demonstrate how MCP transforms natural language requests into reliable, structured tool calls within the domain of CSV data cleaning.

---

## Tool 1: `profile_dataset`

### Purpose
Analyzes a CSV dataset and returns key statistics to support data cleaning decisions.

### Expected Input

```json
{
  "file_path": "sample_dataset.csv"
}
```

### Expected Output

```json
{
  "rows": 5,
  "columns": ["age", "salary"],
  "missing_percentage": {
    "age": 0.2,
    "salary": 0.2
  },
  "duplicates": 1
}
```

### Example Call

```
POST /profile_dataset
```

```json
{
  "file_path": "sample_dataset.csv"
}
```

---

## Tool 2: `remove_duplicates`

### Purpose
Removes duplicate rows from a CSV dataset and saves the cleaned version.

### Expected Input

```json
{
  "file_path": "sample_dataset.csv"
}
```

### Expected Output

```json
{
  "status": "success",
  "rows_removed": 1
}
```

### Example Call

```
POST /remove_duplicates
```

```json
{
  "file_path": "sample_dataset.csv"
}
```

---

## Tool 3: `impute_missing_values`

### Purpose
Fills missing values in a specified column using a chosen imputation strategy (`mean`, `median`, or `mode`).

### Expected Input

```json
{
  "file_path": "sample_dataset.csv",
  "column": "salary",
  "strategy": "mean"
}
```

---

## Tool 4: `drop_rows_with_nulls`

### Purpose
Drops rows that contain missing values. This can be applied to all columns or limited to selected columns.

### Expected Input

```json
{
  "file_path": "sample_dataset.csv",
  "columns": ["salary", "age"]
}
```

If `columns` is omitted or empty, rows with nulls in any column are dropped.

### Expected Output

```json
{
  "status": "success",
  "mode": "selected_columns",
  "columns": ["salary", "age"],
  "rows_before": 100,
  "rows_after": 82,
  "rows_removed": 18,
  "output_file_path": "sample_dataset_cleaned_drop_null_rows.csv"
}
```

### Example Call

```
POST /drop_rows_with_nulls
```

```json
{
  "file_path": "sample_dataset.csv",
  "columns": ["salary", "age"]
}
```

### Expected Output

```json
{
  "status": "success",
  "column": "salary",
  "strategy": "mean",
  "imputed_value": 55000.0,
  "missing_before": 1,
  "missing_after": 0
}
```

### Example Call

```
POST /impute_missing_values
```

```json
{
  "file_path": "sample_dataset.csv",
  "column": "salary",
  "strategy": "mean"
}
```

---

# Prompt-to-Tool Examples

This section demonstrates how natural language requests map to structured MCP tool calls.

---

## Example 1 — Dataset Profiling

### Natural Language Prompt

> "Analyze the dataset and tell me how many missing values it contains."

### Mapped Tool Call

**Tool:** `profile_dataset`

```json
{
  "file_path": "sample_dataset.csv"
}
```

---

## Example 2 — Removing Duplicates

### Natural Language Prompt

> "Remove duplicate rows from the dataset."

### Mapped Tool Call

**Tool:** `remove_duplicates`

```json
{
  "file_path": "sample_dataset.csv"
}
```

---

## Example 3 — Imputing Missing Values

### Natural Language Prompt

> "Fill missing salary values using the mean."

### Mapped Tool Call

**Tool:** `impute_missing_values`

```json
{
  "file_path": "sample_dataset.csv",
  "column": "salary",
  "strategy": "mean"
}
```

---

## Example 4 — Drop Rows With Nulls

### Natural Language Prompt

> "Delete rows where salary is missing."

### Mapped Tool Call

**Tool:** `drop_rows_with_nulls`

```json
{
  "file_path": "sample_dataset.csv",
  "columns": ["salary"]
}
```

---

# Summary

This MCP server exposes three structured tools within the domain of CSV data cleaning. Each tool defines:

- A clear purpose
- A defined input schema
- A structured output format
- Realistic error handling
- Direct mapping from natural language to tool invocation

The use of structured tool calls improves reliability compared to plain prompt-based tool usage by enforcing validation, predictable outputs, and consistent behavior.