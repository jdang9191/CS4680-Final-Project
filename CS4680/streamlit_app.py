#!/usr/bin/env python3
"""Simple Streamlit GUI for the CSV cleaning agent.

Run:
    /Users/Jonny/CS4680/venv/bin/python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime

import streamlit as st

from csv_two_agent import CleanRequest, clean


st.set_page_config(page_title="CSV Cleaning Agent", page_icon="CSV", layout="centered")

st.title("CSV Cleaning Agent")
st.write(
    "Upload a CSV, enter a cleaning prompt, run the agent, and download the outputs."
)
st.info("Limit: 50,000 rows max per file.")

if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = ""
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="CSV files up to 50,000 rows are supported.",
)
prompt = st.text_area(
    "Cleaning prompt",
    placeholder=(
        "Example: Clean this dataset. PromoCode is important, do not drop it. "
        "If Transaction Date is empty replace with None."
    ),
    height=120,
    key="prompt_text",
)

run_clicked = st.button("Run cleaning", type="primary", use_container_width=True)

if run_clicked:
    if uploaded_file is None:
        st.error("Please upload a CSV file.")
    elif not prompt.strip():
        st.error("Please enter a prompt.")
    else:
        with st.spinner("Running cleaner agent..."):
            temp_dir = tempfile.mkdtemp(prefix="csv_clean_gui_")
            input_path = os.path.join(temp_dir, uploaded_file.name)

            try:
                file_bytes = uploaded_file.getvalue()
                with open(input_path, "wb") as input_file:
                    input_file.write(file_bytes)

                req = CleanRequest(file_path=input_path, prompt=prompt)
                result = clean(req)
                output = result.model_dump()

                cleaned_path = output["cleaned_file_path"]
                report_path = output["report_file_path"]

                with open(cleaned_path, "rb") as cleaned_file:
                    cleaned_bytes = cleaned_file.read()
                with open(report_path, "rb") as report_file:
                    report_bytes = report_file.read()
                report_text = report_bytes.decode("utf-8", errors="replace")

                output_json = json.dumps(output, indent=2, ensure_ascii=True).encode("utf-8")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                before = output.get("before_profile", {})
                after = output.get("after_profile", {})
                missing_before = int(sum((before.get("missing_counts", {}) or {}).values()))
                missing_after = int(sum((after.get("missing_counts", {}) or {}).values()))

                st.session_state.latest_result = {
                    "cleaned_bytes": cleaned_bytes,
                    "report_bytes": report_bytes,
                    "report_text": report_text,
                    "output_json": output_json,
                    "timestamp": timestamp,
                    "before": before,
                    "after": after,
                    "missing_before": missing_before,
                    "missing_after": missing_after,
                }

                st.success("Cleaning completed successfully.")
                st.write("### Run Summary")
                st.write(
                    f"Rows: {before.get('rows', '<n/a>')} -> {after.get('rows', '<n/a>')}  ")
                st.write(
                    f"Duplicates: {before.get('duplicates', '<n/a>')} -> {after.get('duplicates', '<n/a>')}  ")
                st.write(f"Missing cells: {missing_before} -> {missing_after}")

                st.write("### Downloads")
                st.download_button(
                    label="Download cleaned CSV",
                    data=cleaned_bytes,
                    file_name=f"cleaned_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"download_csv_{timestamp}",
                )
                st.download_button(
                    label="Download report TXT",
                    data=report_bytes,
                    file_name=f"cleaning_report_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key=f"download_txt_{timestamp}",
                )
                st.download_button(
                    label="Download full JSON output",
                    data=output_json,
                    file_name=f"cleaning_output_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"download_json_{timestamp}",
                )

                with st.expander("View full JSON output"):
                    st.code(output_json.decode("utf-8"), language="json")

                with st.expander("View cleaning report"):
                    st.text(report_text)

            except Exception as exc:
                st.error(f"Cleaning failed: {exc}")
            finally:
                # Keep only in-memory output bytes for downloads and remove temp artifacts.
                for root, dirs, files in os.walk(temp_dir, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except OSError:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except OSError:
                            pass
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass

if st.session_state.latest_result:
    latest = st.session_state.latest_result
    st.write("### Saved Downloads")
    st.download_button(
        label="Download cleaned CSV",
        data=latest["cleaned_bytes"],
        file_name=f"cleaned_{latest['timestamp']}.csv",
        mime="text/csv",
        use_container_width=True,
        key="saved_download_csv",
    )
    st.download_button(
        label="Download report TXT",
        data=latest["report_bytes"],
        file_name=f"cleaning_report_{latest['timestamp']}.txt",
        mime="text/plain",
        use_container_width=True,
        key="saved_download_txt",
    )
    st.download_button(
        label="Download full JSON output",
        data=latest["output_json"],
        file_name=f"cleaning_output_{latest['timestamp']}.json",
        mime="application/json",
        use_container_width=True,
        key="saved_download_json",
    )

    with st.expander("Last run JSON output"):
        st.code(latest["output_json"].decode("utf-8"), language="json")

    with st.expander("Last run cleaning report"):
        st.text(latest["report_text"])
