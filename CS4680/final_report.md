





CSV Cleaning AI Agent
Jonathan Dang
April 21, 2026
California State Polytechnic University - Pomona
CS4280.01
Professor Amamra



















Table of Contents
Introduction…………………………………………………………………….3
Project Overview…………………………….………………………………....4
System Architecture……………………………………………………………5
Prompt Engineering Strategy………………………………………………......5
Implementation………………………………...………………………………7
Case Studies/Examples…………………………………………..………..….10
Discussion…………………………………………………………………….12
Conclusion……………………………………………………………………13
Limitations and Future Work…………………………………………………13



















Executive Summary
This project builds a CSV cleaning system that uses an LLM for planning, but keeps execution deterministic through schema-constrained tools. I designed it as a CLI-first pipeline that can also run through a FastAPI endpoint and Streamlit GUI for file upload.download workflows. The central design decision was to avoid giving the model direct authority over data mutation. Instead, the model proposes a structured plan, guardrails clean up or reject unsafe actions, and only validated tool calls are executed. This report explains the architecture, prompt strategy, implementation detail, and benchmark results from three case studies.

Introduction
Large language models are useful for helping to interpret vague instructions, but they are not reliable enough to be trusted as direct executors for data cleaning. In this project, I treated prompt engineering as a planning interface rather than an execution mechanism. The planner can reason about user intent and dataset statistics while the tool layer enforces concrete rules about what operations are actually allowed.
	That split matters for reproducibility. If a model is allowed to produce arbitrary free-form cleaning behavior, the same task can produce inconsistent results and weak auditability. By forcing the planner to produce structured JSON actions and by constraining execution to typed tools, I can inspect every decision and replay runs with the same inputs.

Motivation: Why Static Prompts Fail
Static prompts failed in early testing for a simple reason: CSV quality problems are not uniform across files. One file may be dominated by duplicates, while another has sparse missingness in many columns, and another has severe missingness in one critical field. A single prompt pattern is not enough because the model needs runtime context to decide whether to deduplicate, impute, or drop.

Another issue is user constraints. In realistic usage, users ask for conditions such as “do not drop salary” or “if Transaction Data is empty, replace it with None.” Static prompting alone can suggest this behavior, but it cannot enforce it. To make those constraints reliable, the project uses explicit prompt parsing and policy checks during plan normalization and execution.

Problem Statement
The project goal is to build a CSV cleaning agent that can accept natural-language requests and still produce reliable, auditable outcomes. More specifically, the system must:	
Accept a CSV path and a prompt
Generate a structured cleaning plan from actual profile metrics
Apply only validated operations with deterministic tools
Return machine-readable and human-readable evidence of what changed.

Approach and Architecture Choice
The final architecture is a tool-augmented, multi-role pipeline. It is not a fully free-form chatbot, and it is not a pure rules engine. The LLM planner contributes flexibility in choosing actions from context, while the tool layer contributes strictness and repeatability. 

I chose this architecture for three practical reasons. First, it handles variability in prompt wording better than static rule logic alone. Second, it limits model failures because invalid actions are normalized or rejected during execution. Third, it generates evidence artifacts automatically, which made debugging and benchmark comparison much easier. 

The system does not currently use vector retrieval. Instead, it loads a local reasoning guide from ‘csv_cleaning_reasoning_framework.txt’ and injects that text into the planner context.

2. Project Overview

End-to-End System Behavior
At runtime, the system starts by profiling the input CSV with ‘profile_dataset’.  This returns dataset shape, inferred data types, missing counts, duplicate statistics, numeric summaries, and cardinality. The planner then receives four inputs together: the user prompt, the profile, parsed prompt constraints, and the reasoning framework text.

The planner returns JSON actions with one of four supported operation types: ‘remove_duplicates’, ‘impute_missing’, ‘drop_columns’, or ‘drop_rows_with_nulls’. That output is not executed immediately. It is normalized first. Normalization removes invalid columns, corrects incompatible strategy choices, blocks protected-column drops, and forces explicit constant values when constant imputation is selected. 

Execution then applies each action through tools in ‘mcp_csv_server/mcp_server.py’. After each pass, the cleaned file is re-profiled. The pipeline can run up to three passes, which helps when early steps (especially deduplication) significantly change downstream missingness patterns. 

After cleaning, the system writes:
A cleaned CSV file.
A fixed-format text report.
A full JSON result payload that includes before/after profiles, plan, execution receipts, and quality checks.

In the GUI flow, the same artifacts are also exposed as download buttons in Streamlit, and the text report is rendered directly on-screen.
Example Scenario
Consider the prompt: ‘Clean this dataset. PromoCode is important, do not drop it. If Transaction Date is empty, replace with None.’

In this scenario, the prompt parser marks ‘PromoCode’ as protected and extracts a replacement rule for ‘Transaction Date’. During normalization and execution, any attempt to drop ‘PromoCode’ is blocked. For “Transaction Date’, strategy is forced to constant with the requested value. If the model misses other columns with missing values, ‘_ensure_all_missing_columns_are_addressed’ appends additional imputation or drop actions so the plan remains complete. 
System Classification
	This project is best described as a tool-augmented A2A-style pipeline with structured planner output. The planner and reporter have different responsibilities, and tool execution is deterministic. This is why I classify it as an agentic system with MCP-style communication patterns rather than a standard prompt-only script.
3. System Architecture (Integrated View)
	


3.1 High-Level Design
NEED diagram here
This design intentionally inserts validation boundaries between planning and execution. The model never writes the file directly. Every state change flows through typed tool calls.
3.2 Retrieval-Augmented Generation (RAG)
RAG is not implemented in this version. This is a deliberate scope decision, not an oversight. I focused first on planner-tool reliability and prompt constraint handling.

The current replacement for RAG is static context injection from ‘csv_cleaning_reasoning_framework.txt’. This gives the model a consistent decision framework without requiring a retriever.

If extended in future work, a RAG version would: 
Build embeddings over framework chunks and prior cleaning examples.
Retrieve top-k segments per prompt.
Inject retrieved context with explicit citation fields to improve traceability

3.3 MCP/Tool Interface
The tool interface is implemented in `mcp_csv_server/mcp_server.py` and follows a schema-first pattern similar to MCP-style tool communication. In this project, "MCP-style" means the model side does not execute arbitrary code directly. Instead, it proposes typed actions, and those actions are routed to deterministic tool functions with validated inputs and structured outputs.

How tools are exposed to the model/planner layer:
- Tool inputs are defined as explicit Pydantic schemas: `FileInput`, `ImputeInput`, `DropColumnsInput`, and `DropRowsWithNullsInput`.
- Tool operations are explicit endpoints/functions: `profile_dataset`, `remove_duplicates`, `impute_missing_values`, `drop_columns`, and `drop_rows_with_nulls`.
- The planner is instructed to return a JSON action list using known action types, and orchestration maps each action to one tool call.

Structured inputs/outputs and schema enforcement:
- Each tool validates required fields before mutation (for example, column existence, strategy validity, numeric compatibility for mean/median).
- `_load_csv` standardizes dirty placeholders (`error`, `n/a`, `na`, `null`, `nan`, `missing`) into true nulls while preserving `unknown` as a valid category.
- Each tool returns machine-readable receipts (for example, `rows_before`, `rows_after`, `rows_removed`, `missing_before`, `missing_after`, `output_file_path`) so behavior can be audited post-run.

Prompt design for tool usage:
- The planner prompt is constrained to produce strict JSON actions, not free-form prose.
- The model decides *which* action types to request based on profile statistics plus prompt constraints.
- The orchestrator decides *whether* each request is safe/valid through normalization and policy checks.
- Executed tool outputs are fed back into the shared state and reused for re-profiling, quality audit, and final report generation.

Key idea:
MCP here functions as a standardized context and tool communication layer: profile data and prompt constraints become shared context, tool calls are typed and deterministic, and receipts become the evidence layer for reproducibility.

One important policy choice in the current version is that row-level null dropping is available but not the default fallback behavior. Row dropping can remove too much data if applied broadly, so the safer default remains imputation and selective column handling unless the prompt explicitly asks to remove rows.

3.4 A2A (Agent-to-Agent Architecture)
This project uses an A2A-style separation of responsibilities inside one orchestrated pipeline. It is not a chat between independent external agents; rather, it is role-based coordination where each role receives different inputs and produces different outputs.

Number and roles of agents:
- Planner role: proposes structured cleaning actions from prompt + dataset profile.
- Executor role (deterministic tool layer): applies approved actions through typed tool functions.
- Reporter role: summarizes before/after outcomes and policy compliance in a stable report format.

Communication model:
- Communication is message/state passing, not hidden memory.
- Shared state includes `before_profile`, `prompt_constraints`, `plan`, `execution_results`, and `after_profile`.
- Each stage consumes explicit artifacts from the prior stage, which makes behavior traceable and debuggable.

Prompting differences by role:
- Planner prompt: optimized for strict JSON action planning and constraint awareness (for example, protected columns, replacement rules, row-drop strictness).
- Reporter prompt/function: optimized for explanation and audit formatting based on execution receipts and metric deltas.
- Executor/tool layer: no generative prompting; it is deterministic code with schema checks.

Control flow:
- Sequential flow: profile -> constraint extraction -> planning -> normalization -> execution -> re-profile -> reporting.
- Iterative collaboration: up to three passes are allowed, so after each execution cycle the system can re-plan using updated dataset statistics.
- Safety gates in each cycle prevent malformed or unsafe model output from being executed directly.

In short, A2A in this system means role-specialized collaboration with explicit state handoffs: one role plans, one layer executes deterministically, and one role reports outcomes with audit evidence.

3.5 Data Flow
The full data flow is:
Receive file path and prompt from CLI or API.
Profile dataset and validate basic support constraints.
Parse protected columns and replacement rules from prompt.
Ask the planner for JSON actions.
Normalize and complete the plan.
Execute actions through tools.
Re-profile output and optionally iterate.
Build report sections and write artifacts.

4. Evaluation Setup
4.1 Benchmark Inputs and Evidence Strategy
To evaluate behavior consistently, I used benchmark runs and saved full JSON outputs under ‘mcp_csv_server/test_data/benchmarks/’. Each benchmark captures:
The original profile.
The final profile.
The plan that was selected.
Tool execution receipts.
File paths for cleaned CSV and report output.

This evidence-first setup made it easy to compare behavior across prompts and datasets without relying on memory of terminal output.

4.2 Evaluation Questions
The benchmarks were chosen to answer three concrete questions:
Can the system clean effectively with minimal prompt guidance?
Do prompt constraints actually change execution behavior in a controlled way?
How does the pipeline behave on larger, noisier data with high missingness?

4.3 Constraints and Validation Rules
The system rejects unsupported data early. One concrete guardrail is a row limit (‘MAX_ALLOWED_ROWS = 50,000’). The validator also rejects blank column names and unsupported data types. These checks prevent costly or ambiguous runs and keep behavior predictable. 

5. Prompt Engineering Strategy

5.1 Base Prompt
The planner system prompt is intentionally strict. It defines role, required JSON shape, allowed action types, allowed strategies, and policy language about protected columns. This was one of the most important improvements over early versions.

The prompt explicitly asks for JSON only and blocks markdown/code fences. Even with this instruction, model output is still treated as untrusted until parsed and normalized.

5.2 Prompting with RAG
RAG is not active in this current implementation. Instead, planner context includes static framework text. While this is less dynamic than retrieval, it gave stable guidance and avoided additional engineering complexity late in the project timeline.

5.3 Prompting for Tool Use (MCP)
Planner output must map directly to tool calls. That is why action records include fields such as ‘type’, ‘column’, ‘strategy’,  and ‘fill_value’. This made the transition from natural language to execution explicit and debuggable. 

When the model produces partial or weak plans, normalization logic compensates. For example:
If strategy is missing, defaults are chosen by type.
If numeric strategy is assigned to text data, it is corrected.
If constant strategy has no value, action is dropped.

5.4 Multi-Agent Prompting (A2A)
Prompting is role-specific. The planner prompt is expressive but constrained, while the reporter stage is deterministic and evidence-driven. I originally considered an LLM-based reporter prompt, but fixed-format reporting proved better for reproducibility and grading.

5.5 Iterative Refinement
Early tests exposed three recurrent issues:
- The planner occasionally ignored some columns with missing values.
- Output could be syntactically valid JSON but semantically incomplete.
- Protected-column intent was sometimes inferred too loosely.

To address this, I added:
- Fallback rule-based planning when LLM parsing fails.
- `_ensure_all_missing_columns_are_addressed` to enforce coverage.
- Hard blocking for protected-column drop actions at execution time.
- Prompt parsing for replacement patterns like `if <column> is empty replace with <value>`.

These changes improved consistency more than prompt tuning alone.

6. Implementation
6.1 Technology Stack
Implementation stack:
- Python for orchestration and tooling.
- FastAPI for optional service mode.
- Pydantic for schema validation.
- Pandas for CSV operations.
- Requests for LLM/API calls.
- Ollama local endpoint for planner model invocation.


6.2 Key Modules
`run_clean.py` is the CLI wrapper. It supports direct mode (local function call) and API mode (`/clean` endpoint). It prints a concise run summary and can save full JSON output.

`csv_two_agent.py` is the orchestrator. It handles profiling, planning, normalization, multi-pass execution, re-profiling, and report generation. It also resolves output directories so files from `test_data/inputs` are written to `test_data/outputs`.

`mcp_csv_server/mcp_server.py` contains deterministic data tools and CSV support logic, including token normalization, dtype-aware imputation checks, row-level null dropping, and output path management.

`streamlit_app.py` provides a lightweight GUI where users upload a CSV, enter a prompt, run cleaning, view summary metrics and the text report, and download cleaned CSV/report/JSON outputs.

6.3 Operational Modes
Direct mode is the default and was used for most benchmarks because it removes API overhead during debugging. API mode is useful when demonstrating service deployment and endpoint behavior. GUI mode (Streamlit) is useful for presentation, since it removes path-typing friction and makes outputs immediately downloadable.

7. Examples and Case Studies

Case Study 1: Baseline Cleaning
Evidence: `mcp_csv_server/test_data/benchmarks/benchmark_1_baseline.json`

Setup:
- Dataset: `sample_dataset_100.csv`
- Prompt: `Clean this dataset.`

Observed metrics:
- Rows: `100 -> 20`
- Duplicates: `80 -> 0`
- Missing cells: `30 -> 0`

Analysis:
This case shows baseline competence without strong prompt constraints. The pipeline removed duplicates first and then imputed missing values. The important point is that this behavior was not only prompt-driven; it was reinforced by fallback and coverage logic. Even if the planner was minimal, the run still converged to a fully cleaned profile.

Case Study 2: Prompt-Constrained Cleaning
Evidence: `mcp_csv_server/test_data/benchmarks/benchmark_2_prompt_constraints.json`

Setup:
- Same base dataset.
- Prompt: `Clean this dataset. salary and name are important and should not be dropped. If color is empty replace with None`

Observed metrics:
- Rows: `100 -> 20`
- Duplicates: `80 -> 0`
- `salary` preserved and imputed.
- `color` receives constant-fill action with `None`.

Analysis:
This case demonstrates controllability. Prompt constraints did not just change planner language; they changed executed behavior. Protected-column semantics carried through normalization and execution. At the same time, global cleaning quality remained strong, which indicates that policy constraints did not break core cleanup behavior.

Case Study 3: Dirty Cafe Dataset
Evidence: `mcp_csv_server/test_data/benchmarks/benchmark_3_dirty_cafe.json`

Setup:
- Dataset: `dirty_cafe_sales_prompt_test.csv` (10,000 rows).
- Prompt: `Clean this dataset. PromoCode is important, do not drop it. If Transaction Date is empty replace with None.`

Observed metrics:
- Rows: `10000 -> 10000`
- Duplicates: `0 -> 0`
- `PromoCode` remains present.
- Multiple missing columns are imputed.
- `Transaction Date` receives constant-fill actions, but post-profile still reports non-zero missing.

Analysis:
This is the most informative case because it highlights both strengths and a real edge case. Policy protection works as intended. Multi-column imputation works at scale. However, literal null handling is tricky. The system maps words like `none` and `null` to a `"None"` literal, and CSV parsing behavior can treat null-like tokens in ways that affect post-profile counts. This does not invalidate the architecture, but it identifies exactly where the next implementation improvement should happen.

8. Limitations
Current limitations are concrete and mostly tied to scope choices:
- No retrieval pipeline. Framework guidance is static rather than query-adaptive.
    This means the model sees the same guidance text for every run, even when only part of that guidance is relevant. It works, but it is less efficient and less targeted than retrieving only the most relevant context for each prompt.
- No dedicated outlier or type-coercion tools, even though reasoning guidance mentions them.
    In practice, this creates a gap between what the reasoning framework can discuss and what the executor can actually do. The planner may reason about coercion or outliers, but the tool layer cannot yet apply those operations directly.
- Null-literal semantics can still cause confusion in some datasets.
    Values like `None`, `null`, and empty strings can be interpreted differently depending on parsing stage and column context. This can produce cases where a replacement action appears to run, but post-profile missing counts do not drop as expected.
- Repeated read/write profiling can increase runtime on larger files.
    The pipeline is intentionally evidence-heavy (profile, execute, re-profile), which improves auditability but adds overhead. This tradeoff is acceptable for project scale, but it matters more on larger datasets.

These boundaries are known and tractable, and most can be improved incrementally without changing the overall architecture.

9. Lessons Learned
The biggest lesson is that prompt engineering alone is not enough for reliable data operations. The biggest gains came from guardrails, fallback planning, and deterministic reporting.

A second lesson is that auditability changes how you debug. Because every run saves plan and execution receipts, it was easier to trace why a result happened and where fixes should go.

A third lesson is that user constraints should be treated as policy objects, not informal text hints. Once protected columns and replacement rules were elevated into explicit constraint structures, behavior became far more consistent.

Another lesson is that the GUI is useful for presentations because it makes the same backend easier to demonstrate without path setup.

10. Conclusion
This project delivers a practical hybrid system for LLM-assisted CSV cleaning. The planner provides flexibility, the tool layer provides reliability, and the report artifacts provide transparency. Benchmark results show strong baseline cleanup and meaningful prompt responsiveness, while also surfacing a specific edge case around null-literal handling. That balance, reliable execution with explicit limitations, is the main outcome of the project.





