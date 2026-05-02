from __future__ import annotations

import json
import os
import shutil
import re
from datetime import datetime
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from mcp_csv_server.mcp_server import (
    DropColumnsInput,
    DropRowsWithNullsInput,
    FileInput,
    ImputeInput,
    drop_columns,
    drop_rows_with_nulls,
    impute_missing_values,
    profile_dataset,
    remove_duplicates,
)

app = FastAPI(title="CSV Two-Agent Cleaning Service")

# This file is the orchestrator. It does not directly edit CSV internals itself.
# Its job is to: profile data, ask the planner for structured actions, enforce
# guardrails, execute typed MCP tools, and generate an auditable report.

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
FRAMEWORK_PATH = os.path.join(os.path.dirname(__file__), "csv_cleaning_reasoning_framework.txt")
MAX_ALLOWED_ROWS = 50_000


class CleanRequest(BaseModel):
    """Request body for the main CSV cleaning endpoint."""
    file_path: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    model_config = ConfigDict(str_strip_whitespace=True)


class CleanAction(BaseModel):
    """One structured action the cleaner agent wants to apply."""
    type: str
    reason: str | None = None
    column: str | None = None
    columns: list[str] | None = None
    strategy: str | None = None
    fill_value: Any | None = None


class CleanPlan(BaseModel):
    """The full cleaning plan returned by the cleaner agent."""
    summary: str | None = None
    actions: list[CleanAction] = []


class CleanResult(BaseModel):
    """What the `/clean` endpoint returns after planning and execution."""
    before_profile: dict[str, Any]
    validation: dict[str, Any]
    cleaned_file_path: str
    report_file_path: str
    plan: CleanPlan
    execution_results: list[dict[str, Any]]
    after_profile: dict[str, Any]
    report: str


class LLMUnavailable(RuntimeError):
    pass


def _load_reasoning_framework() -> str:
    """Load the guidance document the agent should follow when making cleaning decisions."""
    try:
        with open(FRAMEWORK_PATH, "r", encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return ""


def _build_final_output_path(file_path: str) -> str:
    """Create one deterministic output path for the cleaned CSV."""
    directory = _resolve_output_directory(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    if base_name.endswith("_cleaned"):
        return file_path

    return os.path.join(directory, f"{base_name}_cleaned.csv")


def _resolve_output_directory(file_path: str) -> str:
    """Use test_data/outputs when source file is under test_data/inputs, else keep same directory."""
    directory = os.path.dirname(file_path)
    normalized = os.path.normpath(directory)
    marker = os.path.normpath(os.path.join("test_data", "inputs"))

    if normalized.endswith(marker):
        output_dir = os.path.normpath(os.path.join(directory, "..", "outputs"))
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    return directory


def _build_report_output_path(cleaned_file_path: str) -> str:
    """Create a readable text report path next to the cleaned CSV."""
    directory = os.path.dirname(cleaned_file_path)
    base_name = os.path.splitext(os.path.basename(cleaned_file_path))[0]
    date_stamp = datetime.now().strftime("%Y%m%d")
    return os.path.join(directory, f"{base_name}_report_{date_stamp}.txt")


def _write_report_file(report_text: str, cleaned_file_path: str) -> str:
    """Persist the report text to disk and return its path."""
    report_path = _build_report_output_path(cleaned_file_path)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text)
    return report_path


@app.get("/")
def root() -> dict[str, str]:
    """Simple health-style root response."""
    return {"message": "CSV two-agent service is running."}


# This endpoint is the end-to-end control flow. This function proves the architecture:
# inspect first, plan second, execute safely, then audit/report.
@app.post("/clean", response_model=CleanResult)
def clean(request: CleanRequest) -> CleanResult:
    """Main pipeline: profile -> plan -> execute -> re-profile -> explain."""

    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=400, detail="File not found.")

    reasoning_framework = _load_reasoning_framework()

    # First inspect the dataset so the agent has concrete statistics to work with.
    before_profile = profile_dataset(FileInput(file_path=request.file_path))

    # Reject unsupported CSV shapes early so the rest of the pipeline stays predictable.
    validation = _validate_csv_support(before_profile)
    prompt_constraints = _extract_prompt_constraints(request.prompt, before_profile)

    cleaned_file_path, execution_results, final_plan = _run_multi_pass_cleaning(
        request.prompt,
        request.file_path,
        before_profile,
        reasoning_framework,
    )

    # Profile again after cleaning so we can compare before vs. after.
    after_profile = profile_dataset(FileInput(file_path=cleaned_file_path))

    # Ask the reporter agent to turn the raw stats into a human-readable explanation.
    report = _reporter_agent_generate(
        request.prompt,
        before_profile,
        after_profile,
        final_plan,
        execution_results,
        reasoning_framework,
    )
    report = _append_quality_audit(
        report,
        before_profile,
        after_profile,
        execution_results,
        prompt_constraints,
    )
    report_file_path = _write_report_file(report, cleaned_file_path)

    return CleanResult(
        before_profile=before_profile,
        validation=validation,
        cleaned_file_path=cleaned_file_path,
        report_file_path=report_file_path,
        plan=final_plan,
        execution_results=execution_results,
        after_profile=after_profile,
        report=report,
    )


def _run_multi_pass_cleaning(
    user_prompt: str,
    file_path: str,
    initial_profile: dict[str, Any],
    reasoning_framework: str,
    max_passes: int = 3,
) -> tuple[str, list[dict[str, Any]], CleanPlan]:
    """Run multiple planning/execution passes so the agent can fix duplicates first and then revisit the cleaned file."""
    # Presentation note:
    # Multi-pass is a reliability choice. After each pass we re-profile the
    # dataset, so the next plan is based on updated stats rather than stale state.
    current_file_path = file_path
    final_output_path = _build_final_output_path(file_path)
    all_results: list[dict[str, Any]] = []
    final_plan = CleanPlan(summary="No cleaning actions chosen yet.", actions=[])

    current_profile = initial_profile

    for pass_index in range(max_passes):
        prompt_constraints = _extract_prompt_constraints(user_prompt, current_profile)
        plan = _cleaner_agent_plan(
            user_prompt,
            current_profile,
            prompt_constraints,
            reasoning_framework,
        )

        # If dataset has duplicates and plan doesn't already include dedup, add it.
        duplicates_count = int(current_profile.get("duplicates", 0) or 0)
        if duplicates_count > 0 and not any(act.type == "remove_duplicates" for act in plan.actions):
            plan.actions.insert(0, CleanAction(type="remove_duplicates", reason=f"Dataset has {duplicates_count} duplicate rows."))

        # Stop if the agent thinks nothing else is worth doing.
        if not plan.actions:
            final_plan = plan
            break

        current_file_path, pass_results = _execute_plan(
            current_file_path,
            plan,
            current_profile,
            prompt_constraints,
            final_output_path,
        )
        all_results.extend(pass_results)
        final_plan = plan

        # Re-profile the newest cleaned file before the next pass.
        current_profile = profile_dataset(FileInput(file_path=current_file_path))

        # If the dataset is already duplicate-free and has no missing values, stop early.
        if current_profile.get("duplicates", 0) == 0 and sum(current_profile.get("missing_counts", {}).values()) == 0:
            break

    # If the agent chose no actions at all, still create one cleaned copy of the original file.
    if not all_results and not os.path.exists(final_output_path):
        shutil.copyfile(file_path, final_output_path)
        current_file_path = final_output_path

    return current_file_path, all_results, final_plan


def _validate_csv_support(profile: dict[str, Any]) -> dict[str, Any]:
    """Fail fast when the uploaded CSV has unsupported column types or bad headers."""
    row_count = int(profile.get("rows", 0))
    if row_count > MAX_ALLOWED_ROWS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "CSV rejected because it exceeds the configured row limit.",
                "max_allowed_rows": MAX_ALLOWED_ROWS,
                "provided_rows": row_count,
            },
        )

    column_support = profile.get("column_support", {})
    unsupported_columns = [
        column
        for column, info in column_support.items()
        if not isinstance(info, dict) or not info.get("supported", False)
    ]

    blank_columns = [column for column in profile.get("columns", []) if not str(column).strip()]

    if blank_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV rejected because it contains blank column names. "
                "Please provide a header row with real column names."
            ),
        )

    if unsupported_columns:
        details = {
            column: column_support.get(column, {})
            for column in unsupported_columns
        }
        raise HTTPException(
            status_code=400,
            detail={
                "message": "CSV rejected because one or more columns use unsupported types.",
                "supported_column_types": ["numeric", "text", "categorical", "boolean"],
                "unsupported_columns": details,
            },
        )

    return {
        "accepted": True,
        "max_allowed_rows": MAX_ALLOWED_ROWS,
        "provided_rows": row_count,
        "supported_column_types": ["numeric", "text", "categorical", "boolean"],
        "unsupported_columns": [],
    }


def _cleaner_agent_plan(
    user_prompt: str,
    profile: dict[str, Any],
    prompt_constraints: dict[str, Any],
    reasoning_framework: str,
) -> CleanPlan:
    """Use the LLM to decide what cleaning actions should happen."""
    # Presentation note:
    # The LLM is used for planning only. We force a strict JSON schema so the
    # output is machine-checkable before anything is executed.
    system_prompt = (
        "You are the cleaner agent in a two-agent CSV pipeline. "
        "Use the reasoning framework as guidance, not as a rigid checklist. "
        "Always prioritize measurable dataset statistics and the user's prompt constraints. "
        "If the user says a column is important, critical, or must be kept, do not drop it. "
        "Return strict JSON with this shape: "
        '{"summary":"...","actions":[{"type":"remove_duplicates"|"impute_missing"|"drop_columns"|"drop_rows_with_nulls",'
        '"reason":"...","column":"...","columns":["..."],"strategy":"mean"|"median"|"mode"|"constant","fill_value":"..."}]}. '
        "Do not include markdown, code fences, or extra text."
    )

    user_content = json.dumps(
        {
            "user_prompt": user_prompt,
            "dataset_profile": profile,
            "prompt_constraints": prompt_constraints,
            "reasoning_framework": reasoning_framework,
        },
        indent=2,
        default=str,
    )

    try:
        content = _call_llm(system_prompt, user_content)
        plan_data = _extract_json_object(content)
        plan = CleanPlan.model_validate(plan_data)
    except Exception:
        plan = _fallback_plan(user_prompt, profile, prompt_constraints)

    return _normalize_plan(plan, profile, prompt_constraints)


def _execute_plan(
    file_path: str,
    plan: CleanPlan,
    profile: dict[str, Any],
    prompt_constraints: dict[str, Any],
    output_file_path: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Run each planned action using the MCP tool functions and keep the latest CSV path."""
    # Presentation note:
    # This is the policy boundary. Even if the planner asks for risky actions,
    # this layer can refuse them (for example protected-column or broad row-drop
    # violations) and log the refusal in execution receipts.
    results: list[dict[str, Any]] = []
    protected_columns = set(prompt_constraints.get("protected_columns", []))
    strict_any_null_row_drop = bool(prompt_constraints.get("strict_any_null_row_drop", False))
    current_file_path = file_path

    for action in plan.actions:
        # Duplicate removal is a global operation, so no column is needed.
        if action.type == "remove_duplicates":
            try:
                result = remove_duplicates(FileInput(file_path=current_file_path, output_file_path=output_file_path))
                results.append({"action": action.model_dump(), "tool_result": result})
                current_file_path = output_file_path
            except HTTPException as exc:
                results.append({"action": action.model_dump(), "error": getattr(exc, "detail", str(exc))})
            except Exception as exc:
                results.append({"action": action.model_dump(), "error": str(exc)})
            continue

        # Missing-value handling happens one column at a time.
        if action.type == "impute_missing":
            if not action.column or not action.strategy:
                results.append({"action": action.model_dump(), "error": "Missing column or strategy."})
                continue
            try:
                # Normalize planner strategies to tool-acceptable values.
                # Planner may emit strategy="replace" with fill_value="mean" or fill_value="red".
                strategy = (action.strategy or "").lower()
                fill_value = action.fill_value

                if strategy == "replace":
                    # If planner specified a textual fill_value that names a strategy, use it.
                    if isinstance(fill_value, str):
                        fv = fill_value.strip().lower()
                        if fv in ("mean", "median", "mode"):
                            strategy = fv
                            fill_value = None
                        else:
                            # Treat other strings as a constant fill value.
                            strategy = "constant"
                    else:
                        # No explicit fill_value: pick a sensible default based on column support.
                        col_support = profile.get("column_support", {}).get(action.column, {})
                        allowed = col_support.get("allowed_operations", []) if isinstance(col_support, dict) else []
                        if "mean" in allowed:
                            strategy = "mean"
                        elif "mode" in allowed:
                            strategy = "mode"
                        else:
                            strategy = "constant"

                # If strategy is constant but fill_value is missing, that's an error.
                if strategy == "constant" and fill_value is None:
                    results.append({"action": action.model_dump(), "error": "Constant imputation requires a fill_value."})
                    continue

                result = impute_missing_values(
                    ImputeInput(
                        file_path=current_file_path,
                        column=action.column,
                        strategy=strategy,
                        fill_value=fill_value,
                        output_file_path=output_file_path,
                    )
                )
                results.append({"action": action.model_dump(), "tool_result": result})
                current_file_path = output_file_path
            except HTTPException as exc:
                results.append({"action": action.model_dump(), "error": getattr(exc, "detail", str(exc))})
            except Exception as exc:
                results.append({"action": action.model_dump(), "error": str(exc)})
            continue

        # Column dropping is allowed, but protected columns from the prompt are blocked.
        if action.type == "drop_columns":
            if not action.columns:
                results.append({"action": action.model_dump(), "error": "Missing columns."})
                continue

            blocked = sorted(protected_columns.intersection(action.columns))
            if blocked:
                results.append(
                    {
                        "action": action.model_dump(),
                        "error": f"Refused to drop protected columns: {blocked}",
                    }
                )
                continue

            try:
                result = drop_columns(
                    DropColumnsInput(
                        file_path=current_file_path,
                        columns=action.columns,
                        output_file_path=output_file_path,
                    )
                )
                results.append({"action": action.model_dump(), "tool_result": result})
                current_file_path = output_file_path
            except HTTPException as exc:
                results.append({"action": action.model_dump(), "error": getattr(exc, "detail", str(exc))})
            except Exception as exc:
                results.append({"action": action.model_dump(), "error": str(exc)})
            continue

        # Row dropping removes records with nulls in all columns or selected columns.
        if action.type == "drop_rows_with_nulls":
            columns = action.columns if action.columns else None

            if not columns and not strict_any_null_row_drop:
                results.append(
                    {
                        "action": action.model_dump(),
                        "error": (
                            "Refused broad any-column row deletion without strict prompt wording. "
                            "Use explicit wording like 'delete rows with any null in any column'."
                        ),
                    }
                )
                continue

            if columns:
                blocked = sorted(protected_columns.intersection(columns))
                if blocked:
                    results.append(
                        {
                            "action": action.model_dump(),
                            "error": f"Refused to drop rows based on protected columns: {blocked}",
                        }
                    )
                    continue

            result = drop_rows_with_nulls(
                DropRowsWithNullsInput(
                    file_path=current_file_path,
                    columns=columns,
                    output_file_path=output_file_path,
                )
            )
            try:
                results.append({"action": action.model_dump(), "tool_result": result})
                current_file_path = output_file_path
            except HTTPException as exc:
                results.append({"action": action.model_dump(), "error": getattr(exc, "detail", str(exc))})
            except Exception as exc:
                results.append({"action": action.model_dump(), "error": str(exc)})
            continue

        results.append({"action": action.model_dump(), "error": f"Unknown action type: {action.type}"})

    return current_file_path, results


def _reporter_agent_generate(
    user_prompt: str,
    before_profile: dict[str, Any],
    after_profile: dict[str, Any],
    plan: CleanPlan,
    execution_results: list[dict[str, Any]],
    reasoning_framework: str,
) -> str:
    """Generate a deterministic report with stable formatting across datasets."""
    return _build_structured_report(user_prompt, before_profile, after_profile, plan, execution_results)


def _build_structured_report(
    user_prompt: str,
    before_profile: dict[str, Any],
    after_profile: dict[str, Any],
    plan: CleanPlan,
    execution_results: list[dict[str, Any]],
) -> str:
    """Return a concise, fixed-format report so output is easy to compare run-to-run."""
    rows_before = int(before_profile.get("rows", 0) or 0)
    rows_after = int(after_profile.get("rows", 0) or 0)
    missing_before = int(sum((before_profile.get("missing_counts", {}) or {}).values()))
    missing_after = int(sum((after_profile.get("missing_counts", {}) or {}).values()))
    duplicates_before = int(before_profile.get("duplicates", 0) or 0)
    duplicates_after = int(after_profile.get("duplicates", 0) or 0)

    # Build a compact, readable actions list
    actions_lines = _build_column_change_lines(execution_results)
    numbered_actions = []
    for i, line in enumerate(actions_lines, start=1):
        numbered_actions.append(f"{i}. {line.lstrip('- ').strip()}")
    actions_text = "\n".join(numbered_actions)

    date_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = [
        "CSV Cleaning Report",
        "===================",
        f"Date: {date_stamp}",
        f"Prompt: {user_prompt}",
    ]

    overview = [
        "\nDATASET OVERVIEW",
        f"Rows:            {rows_before} -> {rows_after}",
        f"Columns (before): {len(before_profile.get('columns', []))}",
        f"Columns (after):  {len(after_profile.get('columns', []))}",
        f"Missing cells:    {missing_before} -> {missing_after}",
        f"Duplicate rows:   {duplicates_before} -> {duplicates_after}",
    ]

    actions_section = ["\nACTIONS EXECUTED"]
    actions_section.append(actions_text or "- no actions executed")

    plan_summary = ["\nPLAN SUMMARY", f"- {plan.summary or 'No summary provided.'}"]

    return "\n".join(header + overview + actions_section + plan_summary)


def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send a chat request to the local Ollama server."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    if response.status_code != 200:
        raise LLMUnavailable(f"LLM request failed with status {response.status_code}.")

    data = response.json()
    return data["message"]["content"]


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def _normalize_plan(
    plan: CleanPlan,
    profile: dict[str, Any],
    prompt_constraints: dict[str, Any],
) -> CleanPlan:
    """Clean up model output so only valid actions and columns remain."""
    # Presentation note:
    # Normalization turns "maybe-valid" model output into deterministic actions
    # that match real columns, valid strategies, and prompt constraints.
    valid_columns = set(profile.get("columns", []))
    numeric_columns = set(profile.get("numeric_columns", []))
    categorical_columns = set(profile.get("categorical_columns", []))
    protected_columns = set(prompt_constraints.get("protected_columns", []))
    replacement_values = prompt_constraints.get("replacement_values", {})
    impute_constraints = prompt_constraints.get("impute_constraints", {})
    requested_row_drop_columns = set(prompt_constraints.get("drop_rows_with_nulls_columns", []))
    strict_any_null_row_drop = bool(prompt_constraints.get("strict_any_null_row_drop", False))
    
    # Detect conflicts: if a column is both requested for DROP and IMPUTE, the IMPUTE takes precedence.
    # Add such columns to protected_columns to prevent the drop.
    drop_columns_requested = set(prompt_constraints.get("drop_columns_requested", []))
    conflicting_columns = drop_columns_requested.intersection(set(impute_constraints.keys()))
    if conflicting_columns:
        protected_columns.update(conflicting_columns)

    normalized_actions: list[CleanAction] = []
    for action in plan.actions:
        if action.type == "remove_duplicates":
            normalized_actions.append(action)
            continue

        if action.type == "drop_columns":
            # GUARDRAIL: Only allow drop_columns if the user explicitly requested it in the prompt.
            # Prevent the planner from unilaterally deciding to drop columns with a vague prompt.
            if not drop_columns_requested:
                # Planner suggested a drop but user never asked for any columns to be dropped.
                continue
            
            columns = [col for col in (action.columns or []) if col in valid_columns and col not in protected_columns]
            if columns:
                normalized_actions.append(
                    CleanAction(
                        type="drop_columns",
                        columns=columns,
                        reason=action.reason,
                    )
                )
            continue

        if action.type == "drop_rows_with_nulls":
            columns = [col for col in (action.columns or []) if col in valid_columns and col not in protected_columns]

            # If the model omitted columns but the prompt requested specific columns, enforce those.
            if not columns and action.columns is None and requested_row_drop_columns:
                columns = sorted(requested_row_drop_columns)

            # Broad any-column row deletion is only allowed with strict prompt wording.
            if not columns and not strict_any_null_row_drop:
                continue

            normalized_actions.append(
                CleanAction(
                    type="drop_rows_with_nulls",
                    columns=columns if columns else None,
                    reason=action.reason,
                )
            )
            continue

        if action.type == "impute_missing":
            if not action.column or action.column not in valid_columns:
                continue

            # If the prompt explicitly says to drop rows when this column is null,
            # keep that column out of imputation so the row-drop can do its job.
            if action.column in requested_row_drop_columns:
                continue

            strategy = action.strategy
            fill_value = action.fill_value

            # Use impute constraints from prompt if present (e.g., "impute city to Los Angeles")
            if action.column in impute_constraints:
                constraint = impute_constraints[action.column]
                strategy = constraint.get("strategy", "constant")
                fill_value = constraint.get("value")
            # Fallback to general replacement values (e.g., "if X is empty replace with Y")
            elif action.column in replacement_values:
                strategy = "constant"
                fill_value = replacement_values[action.column]

            if not strategy:
                strategy = "median" if action.column in numeric_columns else "mode"

            if strategy in {"mean", "median"} and action.column not in numeric_columns:
                strategy = "mode"
            if strategy == "mode" and action.column in numeric_columns and action.column not in categorical_columns:
                strategy = "median"

            if strategy == "constant" and fill_value is None:
                continue

            normalized_actions.append(
                CleanAction(
                    type="impute_missing",
                    column=action.column,
                    strategy=strategy,
                    fill_value=fill_value,
                    reason=action.reason,
                )
            )

    if not normalized_actions:
        normalized_actions = _fallback_plan("", profile, prompt_constraints).actions

    # Enforce explicit column-drop requests from the user's prompt even if the
    # planner omitted them.
    requested_drop_columns = set(prompt_constraints.get("drop_columns_requested", []))
    if requested_drop_columns:
        # Filter to valid, non-protected columns
        to_drop = [c for c in sorted(requested_drop_columns) if c in valid_columns and c not in protected_columns]
        if to_drop:
            # If not already planned, insert a drop_columns action at the front.
            already_dropped = set()
            for act in normalized_actions:
                if act.type == "drop_columns" and act.columns:
                    already_dropped.update(act.columns)
            missing = [c for c in to_drop if c not in already_dropped]
            if missing:
                normalized_actions.insert(0, CleanAction(type="drop_columns", columns=missing, reason="User requested column drop."))

    if requested_row_drop_columns:
        already_requested_rows = False
        for act in normalized_actions:
            if act.type == "drop_rows_with_nulls" and act.columns:
                if any(column in requested_row_drop_columns for column in act.columns):
                    already_requested_rows = True
                    break
        if not already_requested_rows:
            normalized_actions.insert(
                0,
                CleanAction(
                    type="drop_rows_with_nulls",
                    columns=sorted(requested_row_drop_columns),
                    reason="User requested row drop for null values.",
                ),
            )

    # Reorder actions to match the order specified in the user's prompt.
    action_order = prompt_constraints.get("action_order", [])
    if action_order:
        # Group actions by type
        actions_by_type: dict[str, list[CleanAction]] = {}
        for action in normalized_actions:
            if action.type not in actions_by_type:
                actions_by_type[action.type] = []
            actions_by_type[action.type].append(action)
        
        # Rebuild normalized_actions in the order specified by action_order
        reordered = []
        for action_type in action_order:
            if action_type in actions_by_type:
                reordered.extend(actions_by_type[action_type])
        
        # Append any remaining actions not mentioned in action_order
        for action_type, actions in actions_by_type.items():
            if action_type not in action_order:
                reordered.extend(actions)
        
        normalized_actions = reordered

    # Make sure every missing column is accounted for, even if the model forgets one.
    normalized_actions = _ensure_all_missing_columns_are_addressed(
        normalized_actions,
        profile,
        prompt_constraints,
    )

    return CleanPlan(summary=plan.summary, actions=normalized_actions)


def _ensure_all_missing_columns_are_addressed(
    actions: list[CleanAction],
    profile: dict[str, Any],
    prompt_constraints: dict[str, Any],
) -> list[CleanAction]:
    """Add cleanup actions for any columns that still have missing values."""
    missing_percentage = profile.get("missing_percentage", {})
    numeric_columns = set(profile.get("numeric_columns", []))
    protected_columns = set(prompt_constraints.get("protected_columns", []))
    replacement_values = prompt_constraints.get("replacement_values", {})
    existing_drop_columns = set()
    existing_impute_columns = set()
    existing_rowdrop_columns = set()
    existing_rowdrop_all = False

    for action in actions:
        if action.type == "drop_columns" and action.columns:
            existing_drop_columns.update(action.columns)
        if action.type == "impute_missing" and action.column:
            existing_impute_columns.add(action.column)
        if action.type == "drop_rows_with_nulls":
            if action.columns:
                existing_rowdrop_columns.update(action.columns)
            else:
                existing_rowdrop_all = True

    extra_actions = list(actions)

    for column in profile.get("columns", []):
        missing_rate = float(missing_percentage.get(column, 0.0) or 0.0)
        if missing_rate <= 0:
            continue

        # If the model already planned something for this column, leave it alone.
        if (
            column in existing_drop_columns
            or column in existing_impute_columns
            or column in existing_rowdrop_columns
            or existing_rowdrop_all
        ):
            continue

        # High-missing columns are dropped unless the prompt says they matter.
        if missing_rate >= 0.6 and column not in protected_columns:
            extra_actions.append(
                CleanAction(
                    type="drop_columns",
                    columns=[column],
                    reason=f"{column} has very high missingness ({missing_rate:.0%}).",
                )
            )
            continue

        # Otherwise, impute the missing values using the safest strategy for the type.
        if column in replacement_values:
            strategy = "constant"
            fill_value = replacement_values[column]
        elif column in numeric_columns:
            skew = profile.get("numeric_summary", {}).get(column, {}).get("skew")
            strategy = "median" if skew not in (None, 0.0) and abs(float(skew)) > 0.5 else "mean"
            fill_value = None
        else:
            strategy = "mode"
            fill_value = None

        extra_actions.append(
            CleanAction(
                type="impute_missing",
                column=column,
                strategy=strategy,
                fill_value=fill_value,
                reason=f"{column} still has missing values and should be filled.",
            )
        )

    return extra_actions


def _fallback_plan(user_prompt: str, profile: dict[str, Any], prompt_constraints: dict[str, Any]) -> CleanPlan:
    """Rule-based backup plan used when the LLM output is missing or invalid."""
    # Presentation note:
    # If model output is malformed or unavailable, this rule-based fallback keeps
    # the run usable and prevents total failure.
    actions: list[CleanAction] = []
    missing_percentage = profile.get("missing_percentage", {})
    numeric_summary = profile.get("numeric_summary", {})
    duplicates = int(profile.get("duplicates", 0))
    protected_columns = set(prompt_constraints.get("protected_columns", []))
    requested_row_drop_columns = list(prompt_constraints.get("drop_rows_with_nulls_columns", []))
    strict_any_null_row_drop = bool(prompt_constraints.get("strict_any_null_row_drop", False))

    if duplicates > 0:
        actions.append(
            CleanAction(
                type="remove_duplicates",
                reason=f"The dataset has {duplicates} duplicate rows.",
            )
        )

    if requested_row_drop_columns:
        actions.append(
            CleanAction(
                type="drop_rows_with_nulls",
                columns=requested_row_drop_columns,
                reason="Prompt requests deleting rows when selected columns are missing.",
            )
        )
    elif strict_any_null_row_drop:
        actions.append(
            CleanAction(
                type="drop_rows_with_nulls",
                columns=None,
                reason="Prompt explicitly requests strict any-column null row deletion.",
            )
        )

    for column in profile.get("columns", []):
        missing_rate = float(missing_percentage.get(column, 0.0) or 0.0)
        if missing_rate <= 0:
            continue

        if missing_rate >= 0.6 and column not in protected_columns:
            actions.append(
                CleanAction(
                    type="drop_columns",
                    columns=[column],
                    reason=f"{column} has very high missingness ({missing_rate:.0%}).",
                )
            )
            continue

        if column in protected_columns:
            reason = f"{column} was marked important in the prompt, so it should be preserved."
        else:
            reason = f"{column} contains missing values and can be imputed safely."

        if column in profile.get("numeric_columns", []):
            skew = numeric_summary.get(column, {}).get("skew")
            strategy = "median" if skew not in (None, 0.0) and abs(float(skew)) > 0.5 else "mean"
        else:
            strategy = "mode"

        actions.append(
            CleanAction(
                type="impute_missing",
                column=column,
                strategy=strategy,
                reason=reason,
            )
        )

    if not actions:
        return CleanPlan(summary="No cleaning needed.", actions=[])

    return CleanPlan(summary="Fallback cleaning plan generated from dataset statistics.", actions=actions)


def _extract_prompt_constraints(user_prompt: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Find columns the user explicitly described as important or protected."""
    # Presentation note:
    # This is where natural-language intent becomes enforceable policy:
    # protected columns, explicit replacement values, and row-drop scope.
    columns = list(profile.get("columns", []))

    # Normalize quoted column mentions so patterns like "drop column 'age'" match.
    # Turn single- or double-quoted tokens into unquoted tokens for parsing.
    prompt_unquoted = re.sub(r"[\"']([\w\s\-]+)[\"']", r"\1", user_prompt)
    lowered_prompt = prompt_unquoted.lower()

    protected_markers = [
        "important",
        "critical",
        "keep",
        "must keep",
        "do not drop",
        "don't drop",
        "not drop",
        "preserve",
        "high priority",
    ]

    protected_columns: list[str] = []
    for column in columns:
        if column.lower() in lowered_prompt and any(marker in lowered_prompt for marker in protected_markers):
            protected_columns.append(column)

    replacement_values: dict[str, Any] = {}
    for column in columns:
        # Matches patterns like: "if color is empty replace with None"
        pattern = re.compile(
            rf"if\s+{re.escape(column.lower())}\s+is\s+(?:empty|blank|null|missing)\s+replace\s+with\s+([\w\-\"']+)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(user_prompt)
        if not match:
            continue

        raw_value = match.group(1).strip().strip('"').strip("'")
        replacement_values[column] = _parse_literal_fill_value(raw_value)

    # Use the unquoted prompt for extraction helpers so quoted names match.
    drop_rows_with_nulls_columns = _extract_row_drop_columns(prompt_unquoted, columns)
    drop_columns_requested = _extract_column_drop_requests(prompt_unquoted, columns)
    strict_any_null_row_drop = _is_strict_any_null_row_drop_requested(prompt_unquoted)
    action_order = _extract_action_order_from_prompt(prompt_unquoted)
    impute_constraints = _extract_impute_constraints(prompt_unquoted, columns)

    # Columns with explicit impute requests are protected from being dropped.
    impute_protected = list(impute_constraints.keys())
    all_protected = sorted(set(protected_columns + impute_protected))

    return {
        "protected_columns": all_protected,
        "replacement_values": replacement_values,
        "drop_rows_with_nulls_columns": sorted(set(drop_rows_with_nulls_columns)),
        "drop_columns_requested": sorted(set(drop_columns_requested)),
        "strict_any_null_row_drop": strict_any_null_row_drop,
        "action_order": action_order,
        "impute_constraints": impute_constraints,
    }


def _extract_column_drop_requests(user_prompt: str, columns: list[str]) -> list[str]:
    """Detect prompts requesting explicit column drops (e.g., 'drop column age')."""
    lowered = user_prompt.lower()
    requested: list[str] = []

    for column in columns:
        col_esc = re.escape(column.lower())

        # direct patterns: 'drop column age', 'remove column age', 'delete column age'
        pattern_a = re.compile(rf"\b(?:drop|remove|delete)\s+(?:the\s+)?(?:column\s+)?{col_esc}\b", flags=re.IGNORECASE)

        # 'drop the age column' or "remove the 'age' column"
        pattern_b = re.compile(rf"\b(?:drop|remove|delete)\s+(?:the\s+)?{col_esc}\s+column\b", flags=re.IGNORECASE)

        # plural/CSV list: 'drop columns age, salary' or 'remove columns age and salary'
        pattern_c = re.compile(
            rf"\b(?:drop|remove|delete)\s+columns?\s+[^.\n]*\b{col_esc}\b",
            flags=re.IGNORECASE,
        )

        if pattern_a.search(user_prompt) or pattern_b.search(user_prompt) or pattern_c.search(user_prompt):
            requested.append(column)

    return requested


def _extract_impute_constraints(user_prompt: str, columns: list[str]) -> dict[str, Any]:
    """Extract explicit impute constraints like 'impute null values in city column to Los Angeles'."""
    lowered = user_prompt.lower()
    constraints: dict[str, Any] = {}

    for column in columns:
        col_esc = re.escape(column.lower())

        # Pattern: "impute [nulls/null values/missing] in [column] to [value]"
        pattern = re.compile(
            rf"impute\s+(?:the\s+)?(?:null\s+values?|nulls?|missing)\s+in\s+(?:the\s+)?(?:column\s+)?['\"]?{col_esc}['\"]?(?:\s+column)?\s+to\s+([\w\s\-\"']+)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(user_prompt)
        if match:
            raw_value = match.group(1).strip().strip('"').strip("'")
            constraints[column] = {
                "strategy": "replace",
                "value": _parse_literal_fill_value(raw_value),
            }
            continue

        # Pattern: "replace nulls in [column] to [value]"
        pattern = re.compile(
            rf"replace\s+(?:the\s+)?(?:null\s+values?|nulls?|missing|missing\s+values?)\s+in\s+(?:the\s+)?(?:column\s+)?['\"]?{col_esc}['\"]?(?:\s+column)?\s+to\s+([\w\s\-\"']+)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(user_prompt)
        if match:
            raw_value = match.group(1).strip().strip('"').strip("'")
            constraints[column] = {
                "strategy": "replace",
                "value": _parse_literal_fill_value(raw_value),
            }
            continue

        # Pattern: "fill [column] with [value]"
        pattern = re.compile(
            rf"fill\s+(?:the\s+)?(?:column\s+)?['\"]?{col_esc}['\"]?(?:\s+column)?\s+with\s+([\w\s\-\"']+)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(user_prompt)
        if match:
            raw_value = match.group(1).strip().strip('"').strip("'")
            constraints[column] = {
                "strategy": "replace",
                "value": _parse_literal_fill_value(raw_value),
            }

    return constraints


def _extract_action_order_from_prompt(user_prompt: str) -> list[str]:
    """Extract the order of actions as mentioned in the user prompt."""
    lowered = user_prompt.lower()
    action_order: list[str] = []
    
    # Find all action mentions with their positions
    actions_with_pos: list[tuple[str, int]] = []
    
    # Row removal patterns - match both "rows with null" and "rows where X is null"
    if re.search(r"\b(?:remove|drop|delete)\s+rows?\s+(?:with|where)[^.]*?(?:null|missing)", lowered):
        pos = re.search(r"\b(?:remove|drop|delete)\s+rows?\s+(?:with|where)", lowered).start()
        actions_with_pos.append(("drop_rows_with_nulls", pos))
    
    # Column drop patterns - match "drop column age" or "remove the age column"
    # Pattern 1: drop/remove/delete columns/column (with or without names)
    pattern1 = r"\b(?:drop|remove|delete)\s+columns?\b"
    # Pattern 2: drop/remove/delete [the] <name> column (name followed by word "column")
    pattern2 = r"\b(?:drop|remove|delete)\s+(?:the\s+)?[\w'\"-]+\s+columns?\b"
    
    if re.search(pattern1, lowered) or (re.search(pattern2, lowered) and "rows" not in re.search(pattern2, lowered).group()):
        pos = re.search(r"\b(?:drop|remove|delete)\s+", lowered).start()
        actions_with_pos.append(("drop_columns", pos))
    
    # Deduplication patterns
    if re.search(r"\b(?:remove|drop|delete)\s+(?:all\s+)?duplicates", lowered):
        pos = re.search(r"\b(?:remove|drop|delete)\s+(?:all\s+)?duplicates", lowered).start()
        actions_with_pos.append(("remove_duplicates", pos))
    
    # Imputation/replacement patterns
    if re.search(r"\breplace\s+(?:null|missing|empty)", lowered):
        pos = re.search(r"\breplace\s+(?:null|missing|empty)", lowered).start()
        actions_with_pos.append(("impute_missing", pos))
    
    # Sort by position and return action types
    actions_with_pos.sort(key=lambda x: x[1])
    action_order = [action for action, pos in actions_with_pos]
    
    return action_order


def _extract_row_drop_columns(user_prompt: str, columns: list[str]) -> list[str]:
    """Detect prompts requesting row deletion when specific columns are missing."""
    lowered_prompt = user_prompt.lower()
    requested: list[str] = []

    for column in columns:
        escaped = re.escape(column.lower())

        # Example: "for city, if value is missing, delete the row"
        pattern_a = re.compile(
            rf"(?:for|where)\s+{escaped}(?:\s+column)?[^.\n]*?(?:missing|empty|null)[^.\n]*?(?:delete|drop|remove)[^.\n]*?row",
            flags=re.IGNORECASE,
        )

        # Example: "delete the row if city is missing"
        pattern_b = re.compile(
            rf"(?:delete|drop|remove)[^.\n]*?row[^.\n]*?(?:if|when)[^.\n]*?{escaped}[^.\n]*?(?:missing|empty|null)",
            flags=re.IGNORECASE,
        )

        # Example: "remove rows where age column is null"
        pattern_c = re.compile(
            rf"(?:remove|drop|delete)\s+rows?\s+where[^.\n]*?{escaped}(?:\s+column)?[^.\n]*?(?:missing|empty|null)",
            flags=re.IGNORECASE,
        )

        # Example: "remove rows with null values in age column"
        pattern_d = re.compile(
            rf"(?:remove|drop|delete)\s+rows?\s+with[^.\n]*?(?:null|missing|empty)[^.\n]*?(?:in|for|on)\s+{escaped}(?:\s+column)?",
            flags=re.IGNORECASE,
        )

        if pattern_a.search(user_prompt) or pattern_b.search(user_prompt) or pattern_c.search(user_prompt) or pattern_d.search(user_prompt):
            requested.append(column)
            continue

        # Looser fallback: require both column name and row-delete language in the same prompt.
        if (
            column.lower() in lowered_prompt
            and any(token in lowered_prompt for token in ["delete the row", "drop the row", "remove the row"])
            and any(token in lowered_prompt for token in ["missing", "null", "empty"])
        ):
            requested.append(column)

    return requested


def _is_strict_any_null_row_drop_requested(user_prompt: str) -> bool:
    """Allow broad any-column row deletion only when the prompt is explicit."""
    lowered = user_prompt.lower()
    strict_markers = [
        "any column",
        "any null",
        "any missing",
        "every null",
        "every missing",
        "delete rows with any null",
        "drop rows with any null",
        "remove rows with any null",
        "delete rows with any missing",
        "drop rows with any missing",
        "remove rows with any missing",
        "remove all rows with null",
        "remove all rows with missing",
        "remove rows with null",
        "drop rows with null",
        "delete rows with null",
        "remove rows with missing",
        "drop rows with missing",
        "delete rows with missing",
    ]
    return any(marker in lowered for marker in strict_markers)


def _parse_literal_fill_value(raw_value: str) -> Any:
    """Convert common literal words in prompts into real values."""
    lowered = raw_value.lower()
    if lowered in {"none", "null", "na", "n/a"}:
        return "None"
    return raw_value


def _fallback_report(
    before_profile: dict[str, Any],
    after_profile: dict[str, Any],
    plan: CleanPlan,
    execution_results: list[dict[str, Any]],
) -> str:
    """Simple text report when the reporter LLM call is unavailable."""
    removed_duplicates = next(
        (
            item["tool_result"].get("rows_removed", 0)
            for item in execution_results
            if item.get("action", {}).get("type") == "remove_duplicates" and "tool_result" in item
        ),
        0,
    )

    return (
        "Cleaning summary:\n"
        f"- Rows before: {before_profile.get('rows')}\n"
        f"- Rows after: {after_profile.get('rows')}\n"
        f"- Duplicate rows removed: {removed_duplicates}\n"
        f"- Actions applied: {len(plan.actions)}\n"
        "The cleaner agent selected actions based on dataset statistics, and the reporter agent summarized the effect of those actions."
    )


def _append_quality_audit(
    base_report: str,
    before_profile: dict[str, Any],
    after_profile: dict[str, Any],
    execution_results: list[dict[str, Any]],
    prompt_constraints: dict[str, Any],
) -> str:
    """Append objective quality checks so report consumers can verify the cleaning outcome quickly."""
    # Presentation note:
    # Final report is not just narrative text. This audit block adds measurable
    # before/after quality signals for grading and reproducibility.
    missing_before = int(sum((before_profile.get("missing_counts", {}) or {}).values()))
    missing_after = int(sum((after_profile.get("missing_counts", {}) or {}).values()))
    duplicates_before = int(before_profile.get("duplicates", 0) or 0)
    duplicates_after = int(after_profile.get("duplicates", 0) or 0)
    rows_before = int(before_profile.get("rows", 0) or 0)
    rows_after = int(after_profile.get("rows", 0) or 0)

    missing_reduction = _safe_percent(missing_before - missing_after, missing_before)
    duplicate_reduction = _safe_percent(duplicates_before - duplicates_after, duplicates_before)
    row_retention = _safe_percent(rows_after, rows_before)

    protected_columns = prompt_constraints.get("protected_columns", [])
    after_columns = set(after_profile.get("columns", []))
    dropped_protected = [column for column in protected_columns if column not in after_columns]

    replacement_values: dict[str, Any] = prompt_constraints.get("replacement_values", {})
    replacement_checks: list[str] = []
    for column, fill_value in replacement_values.items():
        matching_results = [
            item for item in execution_results
            if item.get("action", {}).get("type") == "impute_missing"
            and item.get("action", {}).get("column") == column
        ]
        if not matching_results:
            replacement_checks.append(f"- {column}: no matching imputation action executed")
            continue

        last_result = matching_results[-1].get("tool_result", {})
        strategy = str(last_result.get("strategy", ""))
        imputed_value = last_result.get("imputed_value")
        missing_after_column = int(last_result.get("missing_after", 0) or 0)
        status = "PASS" if strategy == "constant" and str(imputed_value) == str(fill_value) and missing_after_column == 0 else "CHECK"
        replacement_checks.append(
            f"- {column}: {status} (strategy={strategy}, imputed_value={imputed_value}, missing_after={missing_after_column})"
        )

    distribution_notes = _build_distribution_notes(before_profile, after_profile)

    action_counts: dict[str, int] = {}
    for item in execution_results:
        action_type = str(item.get("action", {}).get("type", "unknown"))
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

    action_lines = [f"- {name}: {count}" for name, count in sorted(action_counts.items())] or ["- none"]
    replacement_lines = replacement_checks or ["- none"]
    dropped_protected_line = "none" if not dropped_protected else ", ".join(dropped_protected)
    change_lines = _build_column_change_lines(execution_results)
    change_text = "\n".join(change_lines)
    replacement_text = "\n".join(replacement_lines)
    action_text = "\n".join(action_lines)
    distribution_text = "\n".join(distribution_notes)

    # Render audit in aligned plain-text sections for readability
    sep = "\n\n---\n\n"
    receipt = [
        "EXECUTION RECEIPT",
        f"Cleaned rows:     {rows_before} -> {rows_after}",
        f"Missing cells:    {missing_before} -> {missing_after}",
        f"Duplicates:       {duplicates_before} -> {duplicates_after}",
        "Per-column actions:",
        change_text or "- none",
    ]

    quality = [
        "QUALITY AUDIT",
        f"Rows before:      {rows_before}",
        f"Rows after:       {rows_after}",
        f"Row retention:    {row_retention:.2f}%",
        f"Missing before:   {missing_before}",
        f"Missing after:    {missing_after}",
        f"Missing reduced:  {missing_reduction:.2f}%",
        f"Duplicates before: {duplicates_before}",
        f"Duplicates after:  {duplicates_after}",
        f"Duplicates reduced: {duplicate_reduction:.2f}%",
    ]

    policy = [
        "POLICY COMPLIANCE",
        f"Protected columns dropped: {dropped_protected_line}",
        "Prompt replacement checks:",
        replacement_text or "- none",
    ]

    summary = [
        "ACTION SUMMARY",
        action_text or "- none",
        "\nDISTRIBUTION SHIFT CHECK",
        distribution_text or "- no notable shifts",
        "\nMANUAL SPOT-CHECK",
        "Verify 10-20 random rows in the cleaned CSV for semantic correctness.",
    ]

    audit_section = sep + "\n".join(receipt + ["\n"] + quality + ["\n"] + policy + ["\n"] + summary)

    return f"{base_report}{audit_section}"


def _build_column_change_lines(execution_results: list[dict[str, Any]]) -> list[str]:
    """Summarize concrete per-column effects from tool outputs."""
    lines: list[str] = []

    for item in execution_results:
        action = item.get("action", {})
        result = item.get("tool_result", {})
        action_type = action.get("type")

        if action_type == "impute_missing":
            column = action.get("column", "<unknown>")
            strategy = result.get("strategy", action.get("strategy", "<unknown>"))
            imputed_value = result.get("imputed_value", action.get("fill_value"))
            missing_before = result.get("missing_before", "?")
            missing_after = result.get("missing_after", "?")
            lines.append(
                f"- impute {column}: strategy={strategy}, fill={imputed_value}, missing {missing_before} -> {missing_after}"
            )
            continue

        if action_type == "remove_duplicates":
            removed = result.get("rows_removed", "?")
            before = result.get("rows_before", "?")
            after = result.get("rows_after", "?")
            lines.append(f"- deduplicate: rows {before} -> {after}, removed={removed}")
            continue

        if action_type == "drop_columns":
            dropped = result.get("dropped_columns", action.get("columns", []))
            lines.append(f"- drop_columns: {dropped}")
            continue

        if action_type == "drop_rows_with_nulls":
            columns = result.get("columns", action.get("columns", []))
            before = result.get("rows_before", "?")
            after = result.get("rows_after", "?")
            removed = result.get("rows_removed", "?")
            mode = result.get("mode", "any_column")
            lines.append(
                f"- drop_rows_with_nulls ({mode}, columns={columns}): rows {before} -> {after}, removed={removed}"
            )
            continue

        if "error" in item:
            lines.append(f"- {action_type}: error={item.get('error')}")

    return lines or ["- no actions executed"]


def _safe_percent(numerator: int | float, denominator: int | float) -> float:
    """Return a percentage while avoiding divide-by-zero issues."""
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def _build_distribution_notes(before_profile: dict[str, Any], after_profile: dict[str, Any]) -> list[str]:
    """Flag large numeric mean/median shifts that may indicate aggressive cleaning side effects."""
    notes: list[str] = []
    before_summary = before_profile.get("numeric_summary", {}) or {}
    after_summary = after_profile.get("numeric_summary", {}) or {}

    for column, before_stats in before_summary.items():
        after_stats = after_summary.get(column)
        if not isinstance(before_stats, dict) or not isinstance(after_stats, dict):
            continue

        before_mean = before_stats.get("mean")
        after_mean = after_stats.get("mean")
        before_median = before_stats.get("median")
        after_median = after_stats.get("median")

        if before_mean is None or after_mean is None:
            continue

        mean_shift = _safe_percent(abs(float(after_mean) - float(before_mean)), abs(float(before_mean)) or 1.0)
        median_shift = 0.0
        if before_median is not None and after_median is not None:
            median_shift = _safe_percent(abs(float(after_median) - float(before_median)), abs(float(before_median)) or 1.0)

        level = "CHECK" if mean_shift > 20.0 or median_shift > 20.0 else "OK"
        notes.append(
            f"- {column}: {level} (mean_shift={mean_shift:.2f}%, median_shift={median_shift:.2f}%)"
        )

    return notes or ["- no numeric columns to compare"]
