import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from agent.agent import NLQueryAgent

MODELS = [
    "groq/llama-3.3-70b-versatile",
    "groq/llama3-70b-8192",
    "gemini/gemini-1.5-flash",
    "openai/gpt-4o",
    "anthropic/claude-3-5-sonnet-20241022",
]

STAGE_LABELS = {
    "query_understanding":    "Query Understanding",
    "ambiguity_detected":     "Ambiguity Detected",
    "clarification_received": "Clarification Received",
    "query_execution":        "Query Execution",
    "execution_result":       "Execution Result",
    "threshold_refinement":   "Threshold Refinement",
    "visualization_decision": "Visualization Decision",
}


def chat(message, history, agent_state, model):
    agent = agent_state.get("agent")
    if agent is None or agent_state.get("model") != model:
        agent = NLQueryAgent(model=model)
        agent_state["agent"] = agent
        agent_state["model"] = model

    reply = agent.chat(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]

    result_md, plot_img, reasoning_md = "", None, ""

    if (agent.last_result and
            agent.last_result.get("status") == "success" and
            agent.last_result.get("row_count", 0) > 0):

        final = agent.finalize()
        agent.last_result = None
        agent._reasoning = []

        data = final.get("data", [])
        conf = final.get("confidence", 1.0)

        result_md = "### {} | Confidence: {}%\n\n".format(
            final.get("summary", ""), round(conf * 100))

        n_clar = len(final.get("clarifications", []))
        n_ref  = len(final.get("refinements", []))
        if n_clar: result_md += "_{} clarification(s)_  ".format(n_clar)
        if n_ref:  result_md += "_{} refinement(s)_".format(n_ref)
        if n_clar or n_ref: result_md += "\n\n"

        if data:
            keys = list(data[0].keys())
            result_md += "| " + " | ".join(keys) + " |\n"
            result_md += "| " + " | ".join(["---"] * len(keys)) + " |\n"
            for row in data:
                result_md += "| " + " | ".join(
                    str(round(v, 2) if isinstance(v, float) else (v if v is not None else ""))
                    for v in row.values()
                ) + " |\n"

        saved = final.get("_saved_to", {})
        if any(saved.values()):
            result_md += "\n**Saved:** " + " | ".join(
                "`{}: {}`".format(k, v) for k, v in saved.items() if v)

        plot_path = final.get("visualization", {}).get("plot_saved_to")
        if plot_path and os.path.exists(plot_path):
            plot_img = plot_path

        for i, step in enumerate(final.get("chain_of_thought", []), 1):
            label = STAGE_LABELS.get(step.get("stage", ""), step.get("stage", ""))
            reasoning_md += "**[{}] {}**\n\n- **Observed:** {}\n- **Decision:** {}\n\n".format(
                i, label, step.get("observation", ""), step.get("decision", ""))

    return history, agent_state, result_md, plot_img, reasoning_md


def reset(model):
    agent = NLQueryAgent(model=model)
    return [], {"agent": agent, "model": model}, "", None, ""


def submit(message, history, agent_state, model):
    if not message.strip():
        return history, agent_state, gr.update(), gr.update(), gr.update(), ""
    h, s, r, p, rz = chat(message, history, agent_state, model)
    return h, s, r, p, rz, ""


with gr.Blocks(title="NL Financial Query Agent") as demo:
    gr.Markdown(
        "# NL Financial Query Agent\n"
        "Ask financial questions in plain English. "
        "Ambiguous queries trigger multiple-choice clarification before execution."
    )

    agent_state = gr.State({"agent": None, "model": None})

    with gr.Row():
        model_dd  = gr.Dropdown(MODELS, value=MODELS[0], label="Model", scale=3)
        reset_btn = gr.Button("New Session", scale=1)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=500)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="e.g. Show companies with high ROE over the last 3 years",
                    label="Your query", scale=5, lines=1)
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "Show companies with ROE > 20 over the last 3 years",
                    "Rank companies by net_profit_margin descending",
                    "Show revenue trend for top 5 companies",
                    "Companies with high ROCE",
                ],
                inputs=msg_box,
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Results"):
                    result_md = gr.Markdown("_Results will appear here._")
                with gr.Tab("Chart"):
                    plot_out = gr.Image(label="Chart")
                with gr.Tab("Reasoning"):
                    reasoning_md = gr.Markdown("_Reasoning trace will appear here._")

    outs = [chatbot, agent_state, result_md, plot_out, reasoning_md, msg_box]
    send_btn.click(submit, [msg_box, chatbot, agent_state, model_dd], outs)
    msg_box.submit(submit, [msg_box, chatbot, agent_state, model_dd], outs)
    reset_btn.click(reset, [model_dd], [chatbot, agent_state, result_md, plot_out, reasoning_md])


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
