"""
Offline test suite — no API key required.
Tests step 5 (derivability), step 10 (LangGraph graph structure), step 11 (insight + thresholds).
Also validates rate-limit retry logic and history trimming.
Run: python test_flow.py
"""
import json, re, sys, types, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append(condition)


# ── Imports ──────────────────────────────────────────────────────────────────
print("\n[1] Imports")
try:
    from data.loader import DATASET, get_stats, get_columns
    check("data.loader", True, f"shape={DATASET.shape}")
except Exception as e:
    check("data.loader", False, str(e))

try:
    from agent.agent import NLQueryAgent
    from agent.graph import build_graph, run_graph
    from agent.prompts import SYSTEM_PROMPT
    check("agent imports", True)
except Exception as e:
    check("agent imports", False, str(e))

try:
    from tools.validate import validate_query, check_metrics, check_threshold
    check("tools.validate", True)
except Exception as e:
    check("tools.validate", False, str(e))

try:
    from utils.normalize import normalize, should_normalize
    from utils.confidence import compute_confidence
    check("utils", True)
except Exception as e:
    check("utils", False, str(e))


# ── Step 10: LangGraph graph structure ───────────────────────────────────────
print("\n[2] Step 10 — LangGraph")
try:
    g = build_graph()
    nodes = set(g.nodes)
    check("graph compiles", True)
    check("nodes present", {"call_llm", "handle_tools", "handle_text"}.issubset(nodes),
          f"nodes={nodes}")
except Exception as e:
    check("graph compiles", False, str(e))

# Verify chat() delegates to run_graph (not a while-True loop)
import inspect
src = inspect.getsource(NLQueryAgent.chat)
check("chat() uses run_graph", "run_graph" in src)
chat_code_lines = [l for l in src.splitlines() if not l.strip().startswith("#")]
check("chat() no while True", "while True" not in "\n".join(chat_code_lines))


# ── Step 5: LLM derivability — mock LiteLLM ──────────────────────────────────
print("\n[3] Step 5 — LLM derivability check (mocked)")

import agent.agent as _agent_mod
import litellm

_real_completion = litellm.completion

def _mock_completion(**kwargs):
    """Return DERIVABLE for PE, NOT_DERIVABLE for garbage."""
    msgs = kwargs.get("messages", [])
    content = msgs[-1].get("content", "") if msgs else ""
    resp_text = "DERIVABLE: 1 / df['earnings_yield']" if "'PE'" in content else "NOT_DERIVABLE"
    msg = types.SimpleNamespace(content=resp_text, tool_calls=None)
    choice = types.SimpleNamespace(message=msg)
    return types.SimpleNamespace(choices=[choice])

litellm.completion = _mock_completion

from agent.agent import _tool_get_stats
result_pe = json.loads(_tool_get_stats("PE"))
check("PE derivable detected", result_pe.get("derivable") is True, str(result_pe.get("formula", "")))
check("PE formula present", "earnings_yield" in result_pe.get("formula", ""))

result_bad = json.loads(_tool_get_stats("made_up_metric"))
check("unknown metric not derivable", "error" in result_bad)

# Restore
litellm.completion = _real_completion


# ── Step 5: retry on RateLimitError ──────────────────────────────────────────
print("\n[4] Step 5 — derivability retry on RateLimitError (mocked)")

from litellm.exceptions import RateLimitError as _RLE
_call_count = {"n": 0}

def _mock_retry(**kwargs):
    _call_count["n"] += 1
    if _call_count["n"] < 3:
        raise _RLE(message="rate limit", llm_provider="groq", model="x", response=None)
    msg = types.SimpleNamespace(content="NOT_DERIVABLE", tool_calls=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

import time
_real_sleep = time.sleep
time.sleep = lambda s: None   # skip actual waits

litellm.completion = _mock_retry
result_retry = json.loads(_tool_get_stats("made_up_metric2"))
check("retried on RateLimitError", _call_count["n"] == 3, f"calls={_call_count['n']}")
check("graceful after retry", "error" in result_retry)

time.sleep = _real_sleep
litellm.completion = _real_completion


# ── _call_llm retry ───────────────────────────────────────────────────────────
print("\n[5] _call_llm — retry on RateLimitError (mocked)")

_call2 = {"n": 0}
def _mock_llm_rl(**kwargs):
    _call2["n"] += 1
    if _call2["n"] < 3:
        raise _RLE(message="rate limit", llm_provider="groq", model="x", response=None)
    msg = types.SimpleNamespace(content="Hello", tool_calls=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

time.sleep = lambda s: None
litellm.completion = _mock_llm_rl
agent = NLQueryAgent.__new__(NLQueryAgent)
agent.model = "groq/llama-3.3-70b-versatile"
agent.history = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
msg = agent._call_llm()
check("_call_llm retried", _call2["n"] == 3, f"calls={_call2['n']}")
check("_call_llm returned content", msg.content == "Hello")

time.sleep = _real_sleep
litellm.completion = _real_completion


# ── _trim_history ─────────────────────────────────────────────────────────────
print("\n[6] _trim_history — TPM guard")

agent2 = NLQueryAgent.__new__(NLQueryAgent)
agent2.history = [{"role": "system", "content": SYSTEM_PROMPT}]
for i in range(12):
    agent2.history.append({"role": "user",      "content": f"q{i}"})
    agent2.history.append({"role": "assistant",  "content": f"a{i}"})
agent2._trim_history(keep_turns=6)
user_turns = sum(1 for m in agent2.history if m["role"] == "user")
check("trim keeps last 6 user turns", user_turns == 6, f"user_turns={user_turns}")
check("system prompt preserved", agent2.history[0]["role"] == "system")


# ── Step 11: insight + thresholds in finalize() ───────────────────────────────
print("\n[7] Step 11 — insight + thresholds")

agent3 = NLQueryAgent.__new__(NLQueryAgent)
agent3.history = [
    {"role": "system",    "content": SYSTEM_PROMPT},
    {"role": "user",      "content": "companies with ROE > 20"},
    {"role": "assistant", "content": "What threshold?\nA) >15\nB) >20\nC) Custom"},
    {"role": "user",      "content": "B"},
    {"role": "assistant", "content": "These 5 companies show strong ROE above 20, indicating efficient use of equity."},
]
agent3.last_result = {
    "status": "success",
    "result": [{"company": "TCS", "ROE": 25.1}, {"company": "INFY", "ROE": 22.3}],
    "trend_data": [],
    "row_count": 2,
    "code": "df = DATASET.copy()\ndf = df[df['ROE'] > 20]\nresult = df\ntrend_data = []",
    "intent": ["filter"],
    "metrics": ["ROE"],
}
agent3.last_query    = {"intent": ["filter"], "metrics": ["ROE"]}
agent3.clarifications = ["What threshold?"]
agent3.refinements   = []
agent3.overrides     = 0
agent3._reasoning    = []
agent3._plot_info    = {}

final = agent3.finalize()
check("insight extracted",       bool(final.get("insight")), repr(final.get("insight", "")[:60]))
check("insight not MCQ",         "A)" not in final.get("insight", ""))
check("thresholds extracted",    final.get("thresholds") == {"ROE": ">20"}, str(final.get("thresholds")))
check("insight in saved JSON",   "insight" in final)
check("thresholds in saved JSON","thresholds" in final)
check("confidence correct",      final["confidence"] == round(1.0 - 0.1*1, 2), str(final["confidence"]))


# ── Step 12: View distribution first ─────────────────────────────────────────
print("\n[8] Step 12 — View distribution first in refinement MCQ")

from agent.agent import _tool_get_stats as _tgs

# Build an agent that has just received an empty-result refinement MCQ
agent4 = NLQueryAgent.__new__(NLQueryAgent)
agent4.history = [
    {"role": "system",    "content": SYSTEM_PROMPT},
    {"role": "user",      "content": "companies with ROE > 50"},
    {"role": "assistant", "content": (
        "No companies matched ROE > 50.\n"
        "Would you like to:\n"
        "A) Lower threshold to mean\n"
        "B) Lower threshold to median\n"
        "C) View distribution first\n"
        "D) Keep as is"
    )},
]
agent4.last_result  = {"status": "success", "result": [], "trend_data": [], "row_count": 0,
                        "code": "df=DATASET.copy()\ndf=df[df['ROE']>50]\nresult=df\ntrend_data=[]",
                        "intent": ["filter"], "metrics": ["ROE"]}
agent4.last_query   = {"intent": ["filter"], "metrics": ["ROE"]}
agent4.clarifications = []
agent4.refinements  = []
agent4.overrides    = 0
agent4._reasoning   = []
agent4._plot_info   = {}
agent4._pending_tool_loop = False

# Simulate user picking C
reply_c = agent4.chat("C")
check("C returns distribution stats",   "Distribution for ROE" in reply_c, reply_c[:80])
check("C shows Min/Max/Mean/Median",    all(k in reply_c for k in ["Min", "Max", "Mean", "Median"]))
check("C re-asks the MCQ",             "Would you like to" in reply_c or "would you like to" in reply_c.lower())
check("C does not re-execute query",   agent4.last_result.get("row_count") == 0)
check("distribution_viewed logged",    any(s.get("stage") == "distribution_viewed" for s in agent4._reasoning))

# Now simulate user picking A after seeing distribution — should NOT intercept, goes to LLM graph
# (we just verify overrides increments and refinements logged, not that query reruns — no API)
agent4b = NLQueryAgent.__new__(NLQueryAgent)
agent4b.history = [
    {"role": "system",    "content": SYSTEM_PROMPT},
    {"role": "user",      "content": "companies with ROE > 50"},
    {"role": "assistant", "content": (
        "No companies matched ROE > 50.\n"
        "Would you like to:\n"
        "A) Lower threshold to mean\n"
        "B) Lower threshold to median\n"
        "C) View distribution first\n"
        "D) Keep as is"
    )},
]
agent4b.last_result  = {"status": "success", "result": [], "row_count": 0, "trend_data": [],
                         "code": "", "intent": ["filter"], "metrics": ["ROE"]}
agent4b.last_query   = {"intent": ["filter"], "metrics": ["ROE"]}
agent4b.clarifications = []
agent4b.refinements  = []
agent4b.overrides    = 0
agent4b._reasoning   = []
agent4b._plot_info   = {}
agent4b._pending_tool_loop = False

# Patch run_graph to avoid API call
import agent.graph as _gmod
_real_run_graph = _gmod.run_graph
_gmod.run_graph = lambda ag, **kw: "re-executing with lower threshold"
reply_a = agent4b.chat("A")
check("A increments overrides",        agent4b.overrides == 1)
check("A appends to refinements",      len(agent4b.refinements) == 1)
check("A passes to LLM graph",         reply_a == "re-executing with lower threshold")
_gmod.run_graph = _real_run_graph

# Prompt check
check("prompt has 4 options",          all(x in SYSTEM_PROMPT for x in ["A) Lower", "B) Lower", "C) View distribution", "D) Keep as is"]))
check("prompt instructs get_stats on C", "get_stats" in SYSTEM_PROMPT and "C" in SYSTEM_PROMPT)
check("graph suppression covers any refinement MCQ",
      '"Would you like to" in content' in open("agent/graph.py").read())


# ── Summary ───────────────────────────────────────────────────────────────────
total  = len(results)
passed = sum(results)
failed = total - passed
print(f"\n{'='*45}")
print(f"  {passed}/{total} passed" + (f"  |  {failed} FAILED" if failed else "  — all good"))
print(f"{'='*45}\n")
sys.exit(0 if failed == 0 else 1)
