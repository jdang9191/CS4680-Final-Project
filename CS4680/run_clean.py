#!/usr/bin/env python3
"""CLI runner for the two-agent CSV cleaner.

Usage examples:
    python run_clean.py --file mcp_csv_server/test_data/inputs/sample_dataset_100.csv --prompt "Clean this dataset"
    python run_clean.py --file mcp_csv_server/test_data/inputs/sample_dataset_100.csv --prompt "..." --mode api --api-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
from fastapi import HTTPException

from csv_two_agent import CleanRequest, clean


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CSV two-agent cleaner from terminal.")
    parser.add_argument("--file", required=True, help="Path to input CSV file.")
    parser.add_argument("--prompt", required=True, help="Cleaning instruction prompt.")
    parser.add_argument(
        "--mode",
        choices=["direct", "api"],
        default="direct",
        help="direct: call Python function locally, api: call running FastAPI endpoint.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base URL of running API when --mode api is used.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Optional path to save full JSON response.",
    )
    return parser.parse_args()


def _run_direct(file_path: str, prompt: str) -> dict[str, Any]:
    req = CleanRequest(file_path=file_path, prompt=prompt)
    result = clean(req)
    return result.model_dump()


def _run_api(api_url: str, file_path: str, prompt: str) -> dict[str, Any]:
    payload = {"file_path": file_path, "prompt": prompt}
    response = requests.post(f"{api_url.rstrip('/')}/clean", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def _print_summary(result: dict[str, Any]) -> None:
    before = result.get("before_profile", {})
    after = result.get("after_profile", {})

    print("Cleaning completed.")
    print(f"- Cleaned CSV: {result.get('cleaned_file_path', '<missing>')}")
    print(f"- Report file: {result.get('report_file_path', '<missing>')}")
    print(
        f"- Rows: {before.get('rows', '<n/a>')} -> {after.get('rows', '<n/a>')} "
        f"| Duplicates: {before.get('duplicates', '<n/a>')} -> {after.get('duplicates', '<n/a>')}"
    )


if __name__ == "__main__":
    args = _parse_args()
    csv_path = str(Path(args.file).expanduser().resolve())

    try:
        if args.mode == "api":
            output = _run_api(args.api_url, csv_path, args.prompt)
        else:
            output = _run_direct(csv_path, args.prompt)

        _print_summary(output)

        if args.save_json:
            save_path = Path(args.save_json).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
            print(f"- Full JSON saved: {save_path}")

    except HTTPException as exc:
        print(f"Failed: HTTP {exc.status_code} - {exc.detail}")
        raise SystemExit(1) from exc
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        print(f"Failed: API request error - {detail}")
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Failed: {exc}")
        raise SystemExit(1) from exc
