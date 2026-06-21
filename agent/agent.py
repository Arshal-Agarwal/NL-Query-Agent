import json
import litellm
from tools.plot import render_plot
from utils.confidence import compute_confidence
from agent.prompts import SYSTEM_PROMPT

# Set litellm.model in env or pass via MODEL env var.
# Examples:
#   MODEL=gemini/gemini-1.5-flash   + GEMINI_API_KEY
#   MODEL=groq/llama3-70b-8192      + GROQ_API_KEY
#   MODEL=openai/gpt-4o             + OPENAI_API_KEY
#   MODEL=anthropic/claude-3-5-sonnet-20241022 + ANTHROPIC_API_KEY

import os
_OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
DEFAULT_MODEL = os.environ.get("MODEL", "groq/llama-3.3-70b-versatile")

_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "Return min/max/mean/median for a metric. Omit metric (pass empty string) to get full dataset schema instead. Use before run_code for qualitative queries or multi-condition feasibility checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "description": "Metric column name, e.g. 'ROE'"},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute pandas code in a sandbox. DATASET (DataFrame) and pd (pandas) are pre-loaded. Code must assign: result (DataFrame) and trend_data (list of dicts with year key, or []). Start with: df = DATASET.copy()",
            "parameters": {
                "type": "object",
                "properties": {
                    "code":    {"type": "string", "description": "Pandas Python code to execute."},
                    "intent":  {"type": "array", "items": {"type": "string"}, "description": "List of intents: filter, rank, trend"},
                    "metrics": {"type": "array", "items": {"type": "string"}, "description": "Metrics used in this query, e.g. ['ROE', 'PB']"},
                },
                "required": ["code", "intent", "metrics"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_plot",
            "description": "Render and save a chart. plot_type: bar=compare companies, line=trend over time (needs trend_data), scatter=two metrics vs each other. Skip if row_count==1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_type":        {"type": "string", "enum": ["bar", "line", "scatter"]},
                    "y_metric":         {"type": "string", "description": "Metric on Y axis"},
                    "title":            {"type": "string"},
                    "x_metric_scatter": {"type": "string", "description": "Metric on X axis (scatter only)"},
                    "reason":           {"type": "string", "description": "Why this plot type was chosen"},
                },
                "required": ["plot_type", "y_metric", "title", "reason"],
            },
        },
    },
]


import re

_DONT_KNOW = re.compile(
    r"\b(i don'?t know|not sure|no idea|idk|unsure|doesn'?t matter|don'?t care|whatever|no preference)\b",
    re.IGNORECASE,
)

_QUALITATIVE = re.compile(
    r'\b(high|low|good|best|strong|weak|cheap|expensive|profitable|undervalued|overvalued|quality|solid|top|worst|poor)\b',
    re.IGNORECASE
)
_HAS_NUMBER = re.compile(r'[><=!]=?\s*\d+|\d+\s*%')

# Detect when Groq leaks tool calls as plain text instead of structured tool_calls
_LEAKED_TOOL = re.compile(
    r'<function[=/](?:run_code|get_stats|render_plot)[^>]*>|'
    r'"name"\s*:\s*"(?:run_code|get_stats|render_plot)"',
    re.IGNORECASE
)

def _condition_has_number(user_msg: str, metric: str) -> bool:
    """Return True if the user explicitly gave a numeric threshold for this metric."""
    # Look for the metric name followed (nearby) by a comparison+number, or vice versa
    pattern = re.compile(
        rf'(?:{re.escape(metric)}\s*[><=!]=?\s*\d+|\d+\s*%?\s*[><=!]=?\s*{re.escape(metric)})',
        re.IGNORECASE,
    )
    return bool(pattern.search(user_msg)) or bool(_HAS_NUMBER.search(user_msg) and metric.lower() in user_msg.lower() and not _QUALITATIVE.search(user_msg))

def _has_assumed_threshold(user_msg: str, conditions: list) -> bool:
    """Return True if the LLM assumed a numeric threshold for any qualitative word in the query."""
    if not _QUALITATIVE.search(user_msg):
        return False
    for cond in conditions:
        metric = cond.get("metric", "")
        if not _condition_has_number(user_msg, metric):
            return True  # qualitative word present, no explicit number for this metric
    return False


# ── Tool implementations ────────────────────────────────────────────────────

def _tool_get_schema() -> str:
    from data.loader import DATASET
    schema = {
        "columns": list(DATASET.columns),
        "dtypes":  {c: str(DATASET[c].dtype) for c in DATASET.columns},
        "shape":   list(DATASET.shape),
        "sample":  DATASET.head(3).to_dict(orient="records"),
    }
    return json.dumps(schema, default=str)


def _tool_get_stats(metric: str) -> str:
    from data.loader import get_stats, get_columns, DATASET
    import os, litellm as _litellm
    if not metric:
        return json.dumps({
            "columns": list(DATASET.columns),
            "dtypes": {c: str(DATASET[c].dtype) for c in DATASET.columns},
            "shape": list(DATASET.shape),
        })
    cols = get_columns()
    if metric not in cols:
        # Step 5: LLM-driven derivability check
        available = [c for c in cols if c not in ("company", "year")]
        model = os.environ.get("MODEL", "groq/llama-3.3-70b-versatile")
        try:
            resp = _litellm.completion(
                model=model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"The metric '{metric}' is not in the dataset. "
                        f"Available columns: {available}. "
                        "Can this metric be derived from those columns? "
                        "If yes, respond with a single line: DERIVABLE: <pandas expression using df columns>. "
                        "If no, respond with: NOT_DERIVABLE"
                    )
                }],
                temperature=0,
            )
            answer = resp.choices[0].message.content.strip()
        except Exception:
            answer = "NOT_DERIVABLE"

        if answer.startswith("DERIVABLE:"):
            formula = answer[len("DERIVABLE:"):].strip()
            return json.dumps({
                "metric": metric,
                "derivable": True,
                "formula": formula,
                "available_columns": available,
                "message": (
                    f"'{metric}' is not in the dataset but can be derived. "
                    f"Proposed formula: {formula}. "
                    "Ask the user to confirm before using it."
                )
            })
        return json.dumps({
            "error": f"'{metric}' not in dataset and cannot be derived. Available: {available}"
        })
    return json.dumps({"metric": metric, "stats": get_stats(metric)})


def _tool_run_code(code: str, intent: list, metrics: list) -> dict:
    import pandas as pd
    from data.loader import DATASET
    # Strip markdown fences if model adds them
    if code.startswith("```"):
        code = "\n".join(l for l in code.splitlines() if not l.startswith("```")).strip()
    ns = {"DATASET": DATASET, "pd": pd}
    try:
        exec(compile(code, "<run_code>", "exec"), ns)
        result_df  = ns.get("result", pd.DataFrame())
        trend_data = ns.get("trend_data", [])
        return {
            "status":     "success",
            "result":     result_df.to_dict(orient="records"),
            "trend_data": trend_data,
            "row_count":  len(result_df),
            "code":       code,
            "intent":     intent,
            "metrics":    metrics,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "code": code, "row_count": 0}


def _tool_render_plot(last_result: dict, plot_type: str, y_metric: str, title: str,
                      x_metric_scatter: str = None, reason: str = "") -> dict:
    from tools.plot import render_plot as _render
    plot_dec   = {"should_plot": True, "plot_type": plot_type, "reason": reason or "LLM-chosen"}
    result     = last_result.get("result", [])
    trend_data = last_result.get("trend_data", [])
    intent     = last_result.get("intent", [])
    pil_img, path = _render(
        plot_dec, result, trend_data,
        x_metric="company", y_metric=y_metric,
        intent=intent, title=title,
        x_metric_scatter=x_metric_scatter,
    )
    return {"plot_saved_to": path, "plot_type": plot_type, "plot_img": pil_img}


class NLQueryAgent:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.clarifications = []
        self.refinements = []
        self.overrides = 0
        self.last_result = None
        self.last_query = None
        self._plot_info: dict = {}
        self._reasoning: list[dict] = []   # structured CoT log

    def _log(self, stage: str, observation: str, decision: str):
        self._reasoning.append({"stage": stage, "observation": observation, "decision": decision})

    def _call_llm(self, force_tool: bool = False):
        import time
        from litellm.exceptions import RateLimitError, BadRequestError
        # Use "required" when we haven't run any tool yet in this turn (forces proper tool call
        # instead of Groq leaking <function=...> text). Use "auto" once tool results are in
        # history so the LLM can respond in plain text with insight/clarification.
        has_tool_result = any(m.get("role") == "tool" for m in self.history)
        has_assistant_msg = any(m.get("role") == "assistant" for m in self.history)
        # Only force required on the very first turn of a brand-new session.
        # After any assistant message (MCQ, clarification) revert to auto to avoid Groq failures.
        tool_choice = "required" if (force_tool and not has_tool_result and not has_assistant_msg) else "auto"
        for attempt in range(4):
            try:
                return litellm.completion(
                    model=self.model,
                    messages=self.history,
                    tools=_TOOL_SCHEMA,
                    tool_choice=tool_choice,
                ).choices[0].message
            except RateLimitError:
                if attempt == 3:
                    raise
                wait = 2 ** (attempt + 2)
                print(f"[rate limit] retrying in {wait}s...")
                time.sleep(wait)
            except BadRequestError as e:
                if "tool_use_failed" not in str(e) or attempt == 3:
                    raise
                time.sleep(1)

    def _handle_tool_calls(self, msg) -> bool:
        """Dispatch granular tools. Returns True if all tools executed, False if intercepted."""
        self.history.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

            # ── get_schema ──────────────────────────────────────────────────
            if name == "get_schema":
                self._log("schema_inspection", "Agent requested dataset schema.",
                          "Returning columns, dtypes, shape, and sample rows.")
                result_str = _tool_get_schema()

            # ── get_stats ───────────────────────────────────────────────────
            elif name == "get_stats":
                metric = args.get("metric", "")
                self._log("stats_inspection",
                          f"Agent requested stats for '{metric}'.",
                          "Returning min/max/mean/median to inform threshold decision.")
                result_str = _tool_get_stats(metric)

            # ── run_code ────────────────────────────────────────────────────
            elif name == "run_code":
                code    = args.get("code", "")
                intent  = args.get("intent", [])
                metrics = args.get("metrics", [])
                self._log(
                    "query_execution",
                    f"Running pandas code — intent={intent}, metrics={metrics}.",
                    "Executing in sandbox; capturing result and trend_data.",
                )
                run_result = _tool_run_code(code, intent, metrics)
                self.last_result = run_result
                self.last_query  = {"intent": intent, "metrics": metrics}
                row_count = run_result.get("row_count", 0)
                status    = run_result.get("status", "unknown")
                self._log(
                    "execution_result",
                    f"status={status}, row_count={row_count}.",
                    "Proceeding to evaluate result quality." if row_count > 0
                    else "Result is empty — will trigger refinement loop.",
                )
                result_str = json.dumps(
                    {k: v for k, v in run_result.items() if k != "result" or len(v) <= 20},
                    default=str,
                )
                # Trim to minimal context — avoid flooding history with large tables
                cols = list(run_result["result"][0].keys()) if run_result.get("result") else []
                preview_cols = cols[:4]  # only first 4 columns in preview
                preview = [{k: r[k] for k in preview_cols if k in r}
                           for r in run_result.get("result", [])[:3]]
                trimmed = {
                    "status":    run_result["status"],
                    "row_count": row_count,
                    "columns":   cols,
                    "preview":   preview,
                    "trend_data_rows": len(run_result.get("trend_data", [])),
                    "error":     run_result.get("error"),
                }
                result_str = json.dumps(trimmed, default=str)

            # ── render_plot ─────────────────────────────────────────────────
            elif name == "render_plot":
                if not self.last_result or self.last_result.get("row_count", 0) == 0:
                    result_str = json.dumps({"error": "No result to plot. Run run_code first."})
                else:
                    plot_type        = args.get("plot_type", "bar")
                    y_metric         = args.get("y_metric", (self.last_query or {}).get("metrics", ["ROE"])[0])
                    title            = args.get("title", f"Results — {y_metric}")
                    x_metric_scatter = args.get("x_metric_scatter")
                    reason           = args.get("reason", "")
                    self._log(
                        "visualization_decision",
                        f"Agent chose plot_type='{plot_type}', y_metric='{y_metric}', x='{x_metric_scatter}'.",
                        reason or "Rendering chart with normalization applied where appropriate.",
                    )
                    plot_info = _tool_render_plot(self.last_result, plot_type, y_metric, title,
                                                  x_metric_scatter=x_metric_scatter, reason=reason)
                    self._plot_info = plot_info  # stash for finalize()
                    result_str = json.dumps(
                        {"plot_saved_to": plot_info["plot_saved_to"], "plot_type": plot_type},
                        default=str,
                    )

            else:
                result_str = json.dumps({"error": f"Unknown tool: {name}"})

            # Prune: keep only the last 2 tool messages to limit token growth
            tool_msgs = [i for i, m in enumerate(self.history) if m.get("role") == "tool"]
            if len(tool_msgs) >= 2:
                # Remove the oldest tool message and its preceding assistant tool_call message
                oldest = tool_msgs[0]
                # also remove the assistant message that triggered it (one step before)
                remove = sorted({oldest, oldest - 1} & set(range(len(self.history))), reverse=True)
                for idx in remove:
                    if idx >= 0:
                        self.history.pop(idx)
            self.history.append({
                "role":        "tool",
                "tool_call_id": tc.id,
                "content":     result_str,
            })
        return True

    def chat(self, user_message: str) -> str:
        # Detect if this is a refinement response
        is_refinement = any(
            "Would you like to" in (m.get("content") or "") and "Lower to" in (m.get("content") or "")
            for m in self.history[-3:] if m["role"] == "assistant"
        )
        if is_refinement:
            self.overrides += 1
            self.refinements.append(user_message.strip())
            self._log(
                "threshold_refinement",
                f"Empty result triggered refinement MCQ. User chose: '{user_message.strip()}'.",
                "Updating threshold and re-executing query."
            )

        # Detect if this is answering a clarification MCQ
        is_clarification = any(
            any(opt in (m.get("content") or "") for opt in ["A)", "B)", "C)"])
            for m in self.history[-3:] if m["role"] == "assistant"
        ) and not is_refinement

        if is_clarification:
            self._log(
                "clarification_received",
                f"User answered clarification: '{user_message.strip()}'.",
                "Resolving ambiguous terms and thresholds from user response."
            )
            # Block non-answers from reaching the LLM
            if _DONT_KNOW.search(user_message):
                last_clarification = next(
                    (m["content"] for m in reversed(self.history) if m["role"] == "assistant"),
                    ""
                )
                reply = (
                    "I need a specific value to run the query — please pick one of the options above.\n\n"
                    + last_clarification
                )
                self.history.append({"role": "user", "content": user_message})
                self.history.append({"role": "assistant", "content": reply})
                return reply

        self.history.append({"role": "user", "content": user_message})

        # Log the initial query understanding on first user turn
        if len([m for m in self.history if m["role"] == "user"]) == 1:
            self._log(
                "query_understanding",
                f"Received query: '{user_message.strip()}'.",
                "Parsing intent, identifying metrics and time range, detecting ambiguous terms."
            )

        # Step 10: use LangGraph instead of while True loop
        from agent.graph import run_graph
        has_assistant = any(m.get("role") == "assistant" for m in self.history)
        return run_graph(self, first_call=not has_assistant)

    def finalize(self) -> dict:
        if not self.last_result or self.last_result.get("status") != "success":
            return {"status": "no_result"}

        import csv
        from datetime import datetime
        from utils.normalize import should_normalize

        result   = self.last_result.get("result", [])
        trend    = self.last_result.get("trend_data", [])
        intent   = (self.last_query or {}).get("intent", [])
        metrics  = (self.last_query or {}).get("metrics", [])
        y_metric = metrics[0] if metrics else "ROE"

        os.makedirs(_OUTPUTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        chain_of_thought = self._reasoning

        # --- save query code ---
        query_code = self.last_result.get("code", "")
        query_code_path = os.path.join(_OUTPUTS_DIR, f"query_{ts}.py")
        with open(query_code_path, "w") as f:
            f.write("# Auto-generated query code\nfrom data.loader import DATASET\nimport pandas as pd\n\n")
            body = "\n".join(
                l for l in query_code.splitlines()
                if not l.startswith("import ") and not l.startswith("from ")
            )
            f.write(body + "\n")

        # --- plot: use what render_plot tool already produced, or fall back ---
        plot_info = self._plot_info
        plot_path = plot_info.get("plot_saved_to")
        plot_img  = plot_info.get("plot_img")
        plot_type = plot_info.get("plot_type", "bar")
        should_plot = bool(plot_path)

        # Build a plot_dec-compatible dict for the output JSON
        plot_dec = {
            "should_plot": should_plot,
            "plot_type":   plot_type,
            "reason":      "LLM-chosen plot type" if should_plot else "no plot requested",
            "plot_saved_to": plot_path,
            "plot_img":    plot_img,
        }

        # --- save plot code (reproducible standalone script) ---
        plot_code_path = None
        if should_plot:
            do_norm = should_normalize(intent, y_metric)
            plot_code_path = os.path.join(_OUTPUTS_DIR, f"plot_{ts}.py")
            lines = [
                "# Auto-generated plot code",
                "import matplotlib; matplotlib.use('Agg')",
                "import matplotlib.pyplot as plt, json",
                "from utils.normalize import normalize, should_normalize",
                f"with open('outputs/result_{ts}.json') as f: data = json.load(f)",
                "result = data['data']; trend_data = data.get('trend_data', [])",
                f"intent = {intent!r}; y_metric = {y_metric!r}; do_norm = {do_norm!r}",
                "fig, ax = plt.subplots(figsize=(10, 5))",
            ]
            if plot_type == "line":
                lines += [
                    "companies = [k for k in trend_data[0].keys() if k != 'year'] if trend_data else []",
                    "years = [row['year'] for row in trend_data]",
                    "for c in companies:",
                    "    vals = [row.get(c, 0) for row in trend_data]",
                    "    if do_norm: vals = normalize(vals)",
                    "    ax.plot(years, vals, marker='o', label=c)",
                    "ax.set_xlabel('Year'); ax.set_ylabel(y_metric); ax.legend(fontsize=7)",
                ]
            else:
                lines += [
                    "companies = [r['company'] for r in result]",
                    "vals = [r.get(y_metric, 0) for r in result]",
                    "if do_norm: vals = normalize(vals)",
                    "ax.bar(companies, vals)",
                    "ax.set_xlabel('Company'); ax.set_ylabel(y_metric)",
                    "plt.xticks(rotation=30, ha='right')",
                ]
            lines += [
                f"ax.set_title('Results — {y_metric}')",
                "plt.tight_layout()",
                f"plt.savefig('outputs/plot_{ts}.png'); plt.close()",
            ]
            with open(plot_code_path, "w") as f:
                f.write("\n".join(lines) + "\n")

        # --- save CSV ---
        csv_path = None
        if result:
            csv_path = os.path.join(_OUTPUTS_DIR, f"results_{ts}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=result[0].keys())
                writer.writeheader()
                writer.writerows(result)

        final = {
            "status":           "success",
            "summary":          f"{len(result)} companies found",
            "confidence":       compute_confidence(len(self.clarifications), len(self.refinements), self.overrides),
            "clarifications":   self.clarifications,
            "refinements":      self.refinements,
            "chain_of_thought": chain_of_thought,
            "visualization":    plot_dec,
            "data":             result,
            "trend_data":       trend,
        }

        json_path = os.path.join(_OUTPUTS_DIR, f"result_{ts}.json")
        with open(json_path, "w") as f:
            json.dump(final, f, indent=2, default=str)

        final["_saved_to"] = {
            "json":       json_path,
            "csv":        csv_path,
            "query_code": query_code_path,
            "plot_code":  plot_code_path,
        }
        return final


def _print_final(final: dict):
    if final.get("status") == "no_result":
        return

    sep  = "─" * 55
    sep2 = "═" * 55

    data = final.get("data", [])
    if data:
        print(f"\n  Results  ({len(data)} companies)")
        print(f"  {sep}")
        keys = list(data[0].keys())
        col_w = 20
        print("  " + "  ".join(k.ljust(col_w) for k in keys))
        print("  " + "  ".join("-" * col_w for _ in keys))
        for row in data:
            print("  " + "  ".join(
                str(round(v, 2) if isinstance(v, float) else v).ljust(col_w)
                for v in row.values()
            ))
        print(f"  {sep}")

    viz = final.get("visualization", {})
    if viz.get("should_plot") and viz.get("plot_saved_to"):
        print(f"\n  Plot saved  : {viz['plot_saved_to']}  ({viz.get('plot_type')} chart)")

    cot = final.get("chain_of_thought", [])
    if cot:
        print(f"\n  Reasoning Trace")
        print(f"  {sep}")
        stage_labels = {
            "query_understanding":    "Query Understanding",
            "ambiguity_detected":     "Ambiguity Detected",
            "clarification_received": "Clarification Received",
            "query_execution":        "Query Execution",
            "execution_result":       "Execution Result",
            "threshold_refinement":   "Threshold Refinement",
            "visualization_decision": "Visualization Decision",
        }
        for i, step in enumerate(cot, 1):
            label = stage_labels.get(step.get("stage", ""), step.get("stage", ""))
            print(f"\n  [{i}] {label}")
            print(f"      Observed : {step.get('observation', '')}")
            print(f"      Decision : {step.get('decision', '')}")

    saved = final.get("_saved_to", {})
    if any(saved.values()):
        print(f"\n  Saved  : " + "  |  ".join(
            f"{k}: {v}" for k, v in saved.items() if v
        ))

    conf = final.get("confidence")
    if conf is not None:
        print(f"  Confidence : {conf}")
    print()


def run():
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    agent = NLQueryAgent(model=model)

    W = 60
    print(f"\n  {'NL Financial Query Agent':^{W}}")
    print(f"  {'Model: ' + model:^{W}}")
    print(f"  {('─' * W)}")
    print(f"  Type your query below. Ctrl+C or 'exit' to quit.\n")

    while True:
        # prompt
        try:
            user_input = input("\033[1;36m  You\033[0m  ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye.\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("\n  Goodbye.\n")
            raise SystemExit(0)

        print(f"\033[1;32m  Agent\033[0m  ", end="", flush=True)
        response = agent.chat(user_input)
        print(response)

        # auto-finalize whenever a successful result exists
        if agent.last_result and agent.last_result.get("status") == "success" and agent.last_result.get("row_count", 0) > 0:
            final = agent.finalize()
            _print_final(final)
            # reset so we don't re-print on next turn unless a new query runs
            agent.last_result = None
            agent._plot_info  = {}
            agent._reasoning  = []
