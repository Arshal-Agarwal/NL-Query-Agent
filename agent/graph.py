"""
LangGraph-based agent loop (Step 10).
Replaces the while True loop in NLQueryAgent.chat() with an explicit state graph.

Graph nodes:
  call_llm      → call the LLM with current history
  handle_tools  → dispatch tool calls, append results to history
  handle_text   → process plain-text LLM response (MCQ / insight / leaked tool)

Graph edges:
  call_llm  → handle_tools  (if msg has tool_calls)
  call_llm  → handle_text   (if msg is plain text)
  handle_tools → call_llm   (always loop back)
  handle_text  → END        (return to caller)
"""

from __future__ import annotations
import json
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    agent: Any          # NLQueryAgent instance — carries history, last_result, etc.
    output: str         # final text to return to the caller
    first_call: bool    # True only on the very first LLM call of this turn


def _node_call_llm(state: AgentState) -> AgentState:
    agent = state["agent"]
    msg = agent._call_llm(force_tool=state["first_call"])
    agent._last_msg = msg          # stash for routing
    return {**state, "first_call": False}


def _route_after_llm(state: AgentState) -> str:
    msg = state["agent"]._last_msg
    if msg.tool_calls:
        return "handle_tools"
    return "handle_text"


def _node_handle_tools(state: AgentState) -> AgentState:
    agent = state["agent"]
    agent._handle_tool_calls(agent._last_msg)
    return state


def _node_handle_text(state: AgentState) -> AgentState:
    from agent.agent import _LEAKED_TOOL, _DONT_KNOW
    agent = state["agent"]
    msg   = agent._last_msg
    content = msg.content or ""

    # Groq tool-leak: re-inject a correction and signal to loop back
    if _LEAKED_TOOL.search(content):
        agent.history.append({"role": "assistant", "content": content})
        agent.history.append({
            "role": "user",
            "content": "Please use the tool functions directly instead of writing them as text."
        })
        agent._last_msg = agent._call_llm(force_tool=False)
        # recurse once — if it's still leaking we'll just return it
        content = agent._last_msg.content or ""
        if agent._last_msg.tool_calls:
            agent._handle_tool_calls(agent._last_msg)
            # tools ran — loop back to call_llm by returning a sentinel
            agent._pending_tool_loop = True
            return {**state, "output": ""}

    # Suppress spurious refinement MCQ when we already have results
    if (agent.last_result and agent.last_result.get("row_count", 0) > 0
            and "Would you like to" in content and "Lower to" in content):
        content = content[:content.index("Would you like to")].strip()

    agent.history.append({"role": "assistant", "content": content})

    is_refinement_msg = "Would you like to" in content and "Lower to" in content
    if any(opt in content for opt in ["A)", "B)", "C)"]) and not is_refinement_msg:
        agent.clarifications.append(content[:100])
        agent._log(
            "ambiguity_detected",
            "Query contains undefined terms or missing thresholds.",
            "Generating MCQ clarification to resolve before execution."
        )

    return {**state, "output": content}


def _route_after_tools(state: AgentState) -> str:
    # After tools run, always go back to call_llm for the LLM to process results
    return "call_llm"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("call_llm",     _node_call_llm)
    g.add_node("handle_tools", _node_handle_tools)
    g.add_node("handle_text",  _node_handle_text)

    g.set_entry_point("call_llm")
    g.add_conditional_edges("call_llm", _route_after_llm, {
        "handle_tools": "handle_tools",
        "handle_text":  "handle_text",
    })
    g.add_edge("handle_tools", "call_llm")
    g.add_edge("handle_text",  END)

    return g.compile()


_GRAPH = None

def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_graph(agent, first_call: bool = True) -> str:
    """Run the LangGraph loop for one user turn. Returns the agent's text reply."""
    graph = get_graph()
    agent._pending_tool_loop = False
    final_state = graph.invoke({
        "agent":      agent,
        "output":     "",
        "first_call": first_call,
    })
    return final_state["output"]
