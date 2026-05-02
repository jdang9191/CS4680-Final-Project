## Quick CLI Run (No Browser)

Activate the environment:
```bash
source venv/bin/activate
```

Start Ollama in another terminal if needed:
```bash
ollama serve
```

Run cleaner directly (recommended):
```bash
python run_clean.py \
  --file mcp_csv_server/test_data/inputs/sample_dataset_100.csv \
  --prompt "Clean this dataset. salary is important, so do not drop it. If color is empty replace with None"
```

Save full JSON output to file:
```bash
python run_clean.py \
  --file mcp_csv_server/test_data/inputs/sample_dataset_100.csv \
  --prompt "Clean this dataset. salary and name are important and should not be dropped." \
  --save-json mcp_csv_server/test_data/benchmarks/last_clean_output.json
```

## Optional API Mode

Start the FastAPI app:
```bash
uvicorn csv_two_agent:app --reload
```

Run CLI against running API:
```bash
python run_clean.py \
  --mode api \
  --api-url http://127.0.0.1:8000 \
  --file mcp_csv_server/test_data/inputs/sample_dataset_100.csv \
  --prompt "Clean this dataset. salary is important, so do not drop it."
```

## GUI Mode (Upload + Prompt + Download)

Install Streamlit once in the project environment:
```bash
/Users/Jonny/CS4680/venv/bin/pip install streamlit
```

Run the GUI:
```bash
/Users/Jonny/CS4680/venv/bin/python -m streamlit run streamlit_app.py
```

What you can do in the GUI:

- Upload a CSV file from your machine.
- Enter a cleaning prompt.
- Run the same backend pipeline used by `run_clean.py`.
- The app is intended for files up to 50,000 rows.
- Download:
  - cleaned CSV
  - report TXT
  - full JSON output

## Notes

- Test inputs live in: `mcp_csv_server/test_data/inputs/`
- Cleaned CSV is written to: `mcp_csv_server/test_data/outputs/*_cleaned.csv` (when using inputs folder)
- Report text file is written to: `mcp_csv_server/test_data/outputs/*_report_YYYYMMDD.txt` (when using inputs folder)
- Row limit is 50,000 rows.
- GUI page also displays the 50,000-row limit.

## Expected Outputs

After each run you should see:

- `cleaned_file_path` in CLI output/JSON
- `report_file_path` in CLI output/JSON
- Full run JSON if `--save-json` is provided

The report file follows a stable structure:

- CSV Cleaning Report
- Dataset Overview
- Cleaning Actions Executed
- Plan Summary
- Execution Receipt
- Quality Audit

## Final Benchmark Commands

Run these three tests and keep the JSON outputs as evidence:

```bash
/Users/Jonny/CS4680/venv/bin/python run_clean.py \
  --file mcp_csv_server/test_data/inputs/sample_dataset_100.csv \
  --prompt "Clean this dataset." \
  --save-json mcp_csv_server/test_data/benchmarks/benchmark_1_baseline.json
```

```bash
/Users/Jonny/CS4680/venv/bin/python run_clean.py \
  --file mcp_csv_server/test_data/inputs/sample_dataset_100.csv \
  --prompt "Clean this dataset. salary and name are important and should not be dropped. If color is empty replace with None" \
  --save-json mcp_csv_server/test_data/benchmarks/benchmark_2_prompt_constraints.json
```

```bash
/Users/Jonny/CS4680/venv/bin/python run_clean.py \
  --file mcp_csv_server/test_data/inputs/dirty_cafe_sales_prompt_test.csv \
  --prompt "Clean this dataset. PromoCode is important, do not drop it. If Transaction Date is empty replace with None." \
  --save-json mcp_csv_server/test_data/benchmarks/benchmark_3_dirty_cafe.json
```

## Known Limitations

- Numeric inference depends on CSV parse quality; some numeric-like fields may remain categorical strings.
- `UNKNOWN` is treated as a valid literal unless explicitly targeted by prompt instructions.
- Outlier handling and type coercion are framework-level guidance and are not executed by dedicated tools in this version.
- For very large files, runtime increases due to repeated full-file read/write operations.

Datasets to test:
https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training
