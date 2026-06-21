import os, sys, uuid, base64
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent.agent import NLQueryAgent

app = FastAPI(title="NL Financial Query Agent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# In-memory session store: session_id -> NLQueryAgent
_sessions: dict[str, NLQueryAgent] = {}

def _get_agent(session_id: str) -> NLQueryAgent:
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found. Call POST /session/new first.")
    return _sessions[session_id]


class ChatRequest(BaseModel):
    session_id: str
    message: str

class NewSessionRequest(BaseModel):
    model: str | None = None


@app.post("/session/new")
def new_session(req: NewSessionRequest = NewSessionRequest()):
    sid = str(uuid.uuid4())
    model = req.model or os.environ.get("MODEL", "groq/llama-3.3-70b-versatile")
    _sessions[sid] = NLQueryAgent(model=model)
    return {"session_id": sid, "model": model}


@app.post("/chat")
def chat(req: ChatRequest):
    agent = _get_agent(req.session_id)
    reply = agent.chat(req.message)

    # Detect message type
    is_mcq = any(opt in reply for opt in ["A)", "B)", "C)"])
    has_result = (
        agent.last_result is not None
        and agent.last_result.get("status") == "success"
        and agent.last_result.get("row_count", 0) > 0
    )

    result_payload = None
    if has_result:
        final = agent.finalize()
        agent.last_result = None
        agent._plot_info  = {}        agent._reasoning = []

        # Encode plot as base64 if it exists
        plot_b64 = None
        plot_path = final.get("visualization", {}).get("plot_saved_to")
        if plot_path and os.path.exists(plot_path):
            with open(plot_path, "rb") as f:
                plot_b64 = base64.b64encode(f.read()).decode()

        result_payload = {
            "summary":          final.get("summary"),
            "confidence":       final.get("confidence"),
            "clarifications":   final.get("clarifications", []),
            "refinements":      final.get("refinements", []),
            "chain_of_thought": final.get("chain_of_thought", []),
            "visualization":    final.get("visualization", {}),
            "data":             final.get("data", []),
            "trend_data":       final.get("trend_data", []),
            "plot_base64":      plot_b64,
            "saved_to":         final.get("_saved_to", {}),
        }

    return {
        "reply":      reply,
        "is_mcq":     is_mcq,
        "has_result": has_result,
        "result":     result_payload,
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/models")
def list_models():
    return {"models": [
        "groq/llama-3.3-70b-versatile",
        "groq/llama3-70b-8192",
        "gemini/gemini-1.5-flash",
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet-20241022",
    ]}


# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def index():
    return FileResponse("frontend/index.html")
