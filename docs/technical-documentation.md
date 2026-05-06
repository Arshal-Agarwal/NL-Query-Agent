# NL Financial Query Agent — Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Data Layer](#data-layer)
5. [Agent Core](#agent-core)
6. [Tools](#tools)
7. [Utilities](#utilities)
8. [Frontend](#frontend)
9. [Memory & Context Model](#memory--context-model)
10. [Output Format](#output-format)
11. [Configuration](#configuration)
12. [Running the Project](#running-the-project)

---

## Overview

NL Financial Query Agent is a conversational agent that accepts natural language financial queries, resolves ambiguity through structured MCQ clarification, executes pandas queries against a financial dataset, and returns results with visualizations.

**Key design constraints:**
- No hidden assumptions — every ambiguous term is confirmed with the user before execution
- LLM only reasons; data execution is deterministic (pandas `exec` sandbox)
- Full conversation history is replayed on every API call (stateful client, stateless API)
- All outputs are persisted to `outputs/` per run

---

## Architecture

```
User Query
  → NLQueryAgent.chat()
      → Full history sent to LLM (Groq / Gemini / OpenAI / Anthropic via LiteLLM)
      → LLM decides: ask clarification MCQ  OR  call a tool
      → If MCQ: return question to user, wait for answer
      → If tool call: dispatch to one of four tools
          ├── get_schema()       — inspect available columns
          ├── get_stats(metric)  — check distribution before multi-condition queries
          ├── run_code(code)     — execute LLM-written pandas code in sandbox
          └── render_plot(...)   — render and save chart
      → Loop until LLM produces a plain text response (no tool call)
  → NLQueryAgent.finalize()
      → Save result JSON, CSV, query code, plot PNG to outputs/
      → Compute confidence score
      → Return structured result dict
```

---

## Project Structure

```
nl-query-agent/
├── agent/
│   ├── agent.py          # NLQueryAgent — tool dispatch, chat loop, finalize
│   └── prompts.py        # SYSTEM_PROMPT (agent behaviour) + CODEGEN_PROMPT (unused legacy)
├── data/
│   ├── loader.py         # Loads data.xlsx, exposes DATASET, get_columns(), get_stats()
│   ├── data.xlsx         # Financial dataset (~1500 companies, 2015–2025)
│   └── generate_data.py  # Script to regenerate synthetic data.xlsx
├── tools/
│   ├── validate.py       # check_metrics(), check_threshold(), validate_query()
│   ├── execute_query.py  # Legacy execute_query() — kept for backward compatibility
│   └── plot.py           # decide_plot(), render_plot()
├── utils/
│   ├── normalize.py      # normalize(), should_normalize()
│   └── confidence.py     # compute_confidence()
├── app.py                # Gradio 6 frontend
├── main.py               # CLI entry point
├── outputs/              # Auto-created; result_*.json, results_*.csv, query_*.py, plot_*.png
├── .env                  # API keys (not committed)
├── .env.example          # API key template
└── requirements.txt
```

---

## Data Layer

### `data/loader.py`

Loads `data/data.xlsx` at import time into a module-level `DATASET` DataFrame.

**Columns after rename:**

| Column | Type | Description |
|---|---|---|
| `company` | str | Ticker symbol |
| `year` | int | Fiscal year (2015–2025) |
| `ROE` | float | Return on Equity (%) |
| `ROCE` | float | Return on Capital Employed (%) |
| `ROA` | float | Return on Assets (%) |
| `net_profit_margin` | float | Net Profit Margin (%) |
| `EPS` | float | Basic EPS (Rs.) |
| `earnings_yield` | float | Earnings Yield (= 1/PE) |
| `enterprise_value` | float | Enterprise Value (Cr.) |
| `PB` | float | Price-to-Book ratio |
| `price_to_revenue` | float | Price / Net Operating Revenue |
| `revenue_per_share` | float | Revenue from Operations per Share (Rs.) |

**Exposed functions:**
- `get_columns() -> list` — returns all column names
- `get_stats(metric: str) -> dict` — returns `{min, max, mean, median}` for a metric

### `data/generate_data.py`

Generates a synthetic `data.xlsx` with ~1500 rows (130 tickers × 11 years). Run once if the file is missing:

```bash
python data/generate_data.py
```

---

## Agent Core

### `agent/agent.py` — `NLQueryAgent`

The central class. One instance per user session.

**State:**

| Attribute | Type | Purpose |
|---|---|---|
| `model` | str | LiteLLM model string |
| `history` | list[dict] | Full conversation history (system + user + assistant + tool messages) |
| `clarifications` | list[str] | Logged MCQ questions asked (used for confidence scoring) |
| `refinements` | list[str] | Logged refinement answers (empty result loop) |
| `overrides` | int | Count of threshold overrides (penalises confidence) |
| `last_result` | dict | Most recent `run_code` result |
| `last_query` | dict | Most recent query metadata (intent, metrics, code) |
| `_reasoning` | list[dict] | Chain-of-thought log entries |
| `_last_run_*` | various | Shared state between `run_code` and `render_plot` tool calls |

**`chat(user_message) -> str`**

Main entry point. Appends the user message to history, calls the LLM in a loop, dispatches tool calls until the LLM produces a plain text response, then returns it.

Interrupt points (returns early with a clarification question):
1. Qualitative word with no numeric threshold detected → MCQ for threshold
2. Unknown metric in `run_code` call → LLM derivability check → MCQ for confirmation or alternative
3. Multi-condition query with infeasible combination → agent calls `get_stats` first, then MCQ

**`finalize() -> dict`**

Called after a successful `run_code`. Saves all outputs and returns the structured result. Should be called once per completed query, then `last_result` and `_reasoning` reset.

**`_call_llm()`**

Wraps `litellm.completion` with retry logic (4 attempts, exponential backoff) for rate limit and bad request errors.

**`_dispatch_tool(name, args) -> str`**

Routes tool calls to the four internal tool methods. Returns JSON string.

### Tool methods

#### `_tool_get_schema()`
Returns `DATASET` column names and 3 sample rows as JSON. The LLM calls this when unsure what data is available.

#### `_tool_get_stats(metric)`
Calls `data.loader.get_stats()` and logs a `feasibility_check` CoT entry. The LLM is instructed to call this before multi-condition queries to reason about joint feasibility before committing to `run_code`.

#### `_tool_run_code(code, intent, metrics)`
Executes LLM-written pandas code in an isolated namespace `{"DATASET": DATASET, "pd": pd}`. The code must store its result in `result` (DataFrame) and `trend_data` (list of dicts). Updates `last_result`, `last_query`, and the shared `_last_run_*` state for `render_plot`.

Before execution, intercepts calls where any metric in `metrics` is not in the dataset and triggers the LLM derivability check (Step 5).

#### `_tool_render_plot(plot_type, x, y, title)`
Calls `tools/plot.render_plot()` using the data from the last `run_code` call. The LLM chooses `plot_type` ("bar" or "line"), `x` axis field, and `y` metric. Saves the plot to `outputs/plot_*.png`.

### Step 5 — LLM Derivability Check

`_check_derivability(metric, model)` makes a focused zero-temperature LLM call asking whether `metric` can be derived from the available columns. Returns a pandas expression string if derivable, `None` otherwise. The result is shown to the user as an MCQ before any execution proceeds.

### `agent/prompts.py`

**`SYSTEM_PROMPT`** — instructs the agent on:
- Available metrics and dataset shape
- Tool usage sequence (get_schema → get_stats → run_code → render_plot)
- When to ask vs when to execute directly
- MCQ format requirements
- Handling "I don't know" responses
- Post-execution behaviour (insight text, empty result refinement loop)

**`CODEGEN_PROMPT`** — legacy prompt for the old template-based codegen path. Kept for `tools/execute_query.py` backward compatibility.

---

## Tools

### `tools/validate.py`

Stateless validation helpers. Used by the legacy `execute_query` path; also callable independently.

- `check_metrics(metrics) -> dict` — checks all metrics exist in `VALID_METRICS`; returns `{valid, missing, available}`
- `check_threshold(metric, op, value) -> dict` — checks a single condition against dataset min/max; returns feasibility warning if threshold would likely return 0 rows
- `validate_query(q) -> (bool, str)` — combines both checks; returns `(True, "")` or `(False, error_message)`

Note: The hardcoded `DERIVABLE` dict was removed in the current version. Derivability is now handled by the LLM in `agent.py`.

### `tools/execute_query.py`

Legacy path kept for backward compatibility. Accepts a structured query dict, validates it, calls the LLM to generate pandas code via `CODEGEN_PROMPT`, and executes it. Not used by the main agent loop (which now uses `run_code` directly), but still functional.

### `tools/plot.py`

- `decide_plot(intent, row_count) -> dict` — rule-based fallback: "trend" → line, "rank"/other → bar. Used by `finalize()` if `render_plot` tool was not called.
- `render_plot(plot_decision, result, trend_data, x_metric, y_metric, intent, title) -> str` — renders matplotlib figure, saves to `outputs/plot_*.png`, returns path. Calls `should_normalize` to decide whether to normalize values.

---

## Utilities

### `utils/normalize.py`

- `normalize(values) -> list` — min-max normalization to [0, 1]
- `should_normalize(intent, metric) -> bool` — returns `True` for rank/filter intents on non-percentage metrics. Percentage metrics (ROE, ROCE, ROA, net_profit_margin, earnings_yield) and trend charts always use raw values.

### `utils/confidence.py`

- `compute_confidence(clarifications, retries, overrides) -> float` — starts at 1.0, deducts 0.1 per clarification, 0.1 per retry, 0.05 per override. Floor at 0.0.

---

## Frontend

### `app.py` — Gradio 6

Single-file Gradio frontend. One `NLQueryAgent` instance per browser tab stored in `gr.State`.

**Layout:**
- Left column (3/5): Chatbot, text input + Send button, MCQ quick-reply buttons (A/B/C/D), New Session button
- Right column (2/5): Model dropdown, Chart image, Results Table accordion, Reasoning Trace accordion

**MCQ buttons:** After every agent reply, `_parse_mcq_options()` scans for `A) ... B) ... C) ...` lines and renders them as clickable buttons. Clicking submits that option as a user message. Buttons hide after submission.

**Auto-finalize:** After each `chat()` call, if `agent.last_result` has `status=success` and `row_count > 0`, `finalize()` is called automatically. The plot appears inline, the results table populates, and the reasoning trace updates.

**Model switching:** Changing the dropdown creates a fresh `NLQueryAgent` with the new model and clears all state.

**Supported models** (via LiteLLM):

| Model string | Provider | Key env var |
|---|---|---|
| `groq/llama-3.3-70b-versatile` | Groq | `GROQ_API_KEY` |
| `gemini/gemini-1.5-flash` | Google | `GEMINI_API_KEY` |
| `openai/gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `anthropic/claude-3-5-sonnet-20241022` | Anthropic | `ANTHROPIC_API_KEY` |

---

## Memory & Context Model

The agent is **stateful on the client, stateless on the API**.

Every `litellm.completion` call sends the full `self.history` list — system prompt, all user messages, all assistant replies, and all tool results. The API processes this fresh each time and returns the next message.

**Implications:**
- Perfect recall within a session — the LLM sees everything
- Token cost grows linearly with conversation length
- No cross-session memory — restarting the app or clicking "New Session" creates a blank history
- Results are persisted to `outputs/*.json` and can be referenced manually

---

## Output Format

Each completed query saves to `outputs/`:

**`result_YYYYMMDD_HHMMSS.json`**
```json
{
  "status": "success",
  "summary": "12 companies found",
  "confidence": 0.8,
  "clarifications": ["What threshold defines..."],
  "refinements": [],
  "chain_of_thought": [
    {"stage": "query_understanding", "observation": "...", "decision": "..."},
    {"stage": "feasibility_check",   "observation": "...", "decision": "..."},
    {"stage": "query_execution",     "observation": "...", "decision": "..."},
    {"stage": "execution_result",    "observation": "...", "decision": "..."},
    {"stage": "visualization_decision", "observation": "...", "decision": "..."}
  ],
  "visualization": {"should_plot": true, "plot_type": "bar", "plot_saved_to": "outputs/plot_*.png"},
  "data": [{"company": "TCS", "ROE": 38.2, ...}],
  "trend_data": []
}
```

**`results_YYYYMMDD_HHMMSS.csv`** — flat CSV of the result rows

**`query_YYYYMMDD_HHMMSS.py`** — the pandas code that was executed

**`plot_YYYYMMDD_HHMMSS.png`** — bar or line chart

### Chain-of-Thought Stages

| Stage | When logged |
|---|---|
| `query_understanding` | First user message received |
| `feasibility_check` | `get_stats` tool called |
| `ambiguity_detected` | MCQ clarification generated |
| `clarification_received` | User answered an MCQ |
| `query_execution` | `run_code` tool called |
| `execution_result` | `run_code` returned |
| `threshold_refinement` | User answered empty-result refinement MCQ |
| `visualization_decision` | `render_plot` tool called |
| `derivability_check` | Unknown metric intercepted |

---

## Configuration

**`.env`** (copy from `.env.example`):
```
GROQ_API_KEY=your_key_here
# GEMINI_API_KEY=
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
```

**Model override** (CLI or env):
```bash
MODEL=gemini/gemini-1.5-flash python main.py
MODEL=openai/gpt-4o python app.py
```

---

## Running the Project

**Install:**
```bash
pip install -r requirements.txt
```

**Generate data** (if `data/data.xlsx` is missing):
```bash
python data/generate_data.py
```

**Web UI:**
```bash
python app.py
```

**CLI:**
```bash
python main.py
# or with model override:
MODEL=groq/llama-3.3-70b-versatile python main.py
```
