"""HTTP layer: resume upload, one request per spoken answer, report.

The browser only sends audio and plays back questions. Prompts, transcript and grades
live on the server in the LangGraph checkpoint, so the client can't edit its own score.
"""
import logging
import os
import sqlite3
import threading
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

load_dotenv()

import graph  # noqa: E402  (reads LLM_MODEL after .env is loaded)
import voice  # noqa: E402
from resume import extract_profile, fetch_resources, read_resume  # noqa: E402

MAX_UPLOAD = 5 * 1024 * 1024
log = logging.getLogger("prepai")

app = FastAPI(title="PrepAI")
interviews = graph.build_graph(SqliteSaver(sqlite3.connect(os.getenv("DB_PATH", "prepai.db"), check_same_thread=False)))
locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)  # one answer at a time per interview


def config(sid: str) -> dict:
    return {"configurable": {"thread_id": sid}}


def pending_question(result: dict) -> dict | None:
    return result["__interrupt__"][0].value if result.get("__interrupt__") else None


@app.post("/api/interviews")
def start(resume: UploadFile = File(...), role: str = Form(...), questions: int = Form(8)):
    data = resume.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Resume must be under 5 MB")
    if not (resume.filename or "").lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(415, "Upload a PDF, TXT or MD resume")
    role = role.strip()[:80] or "Software Engineer"

    text, links = read_resume(data, resume.filename)
    if len(text) < 200:
        raise HTTPException(422, "Couldn't read text from that resume. Is it a scanned image?")
    resources = fetch_resources(links)
    profile = extract_profile(graph.llm(), text, resources, role)
    if not profile.claims:
        raise HTTPException(422, "Couldn't find any concrete experience to ask about")

    sid = uuid.uuid4().hex
    initial = {"role": role, "profile": profile.model_dump(), "budget": max(3, min(questions, 15)), "turns": []}
    result = interviews.invoke(initial, config(sid))
    return {
        "id": sid, "role": role, "name": profile.name, "budget": initial["budget"],
        "sources": ["resume", *resources], **pending_question(result),
    }


@app.post("/api/interviews/{sid}/answer")
def answer(sid: str, audio: UploadFile | None = File(None), text: str | None = Form(None)):
    state = interviews.get_state(config(sid))
    if not state.values:
        raise HTTPException(404, "Interview not found")
    if not state.next:
        return {"done": True}

    with locks[sid]:
        if text and text.strip():
            said = text.strip()
        elif audio:
            said = voice.transcribe(audio.file.read(MAX_UPLOAD), audio.filename or "answer.webm", state.values["profile"]["keywords"])
        else:
            said = ""
        if len(said) < 2:
            raise HTTPException(422, "Didn't catch that, try again")
        result = interviews.invoke(Command(resume=said), config(sid))

    question = pending_question(result)
    return {"transcript": said, "done": question is None, **(question or {})}


@app.get("/api/interviews/{sid}/speech")
def speech(sid: str):
    """Audio for the question currently waiting on an answer. Only that text, so the endpoint can't voice arbitrary input."""
    state = interviews.get_state(config(sid))
    if not state.next:
        raise HTTPException(404, "No question waiting")
    try:
        audio = voice.synthesize(state.values["question"])
    except Exception as err:  # the browser falls back to its built-in voice
        log.warning("TTS failed: %s", err)
        raise HTTPException(502, "Voice unavailable")
    return Response(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.post("/api/interviews/{sid}/end")
def end(sid: str):
    state = interviews.get_state(config(sid))
    if not state.values:
        raise HTTPException(404, "Interview not found")
    if state.next:
        with locks[sid]:
            interviews.invoke(Command(resume=graph.END_SIGNAL), config(sid))
    return {"done": True}


@app.get("/api/interviews/{sid}")
def result(sid: str):
    state = interviews.get_state(config(sid))
    if not state.values:
        raise HTTPException(404, "Interview not found")
    v = state.values
    return {
        "role": v["role"], "name": v["profile"]["name"], "done": not state.next,
        "score": v.get("score"), "report": v.get("report"),
        "turns": [t for t in v.get("turns", []) if t["answer"] != graph.END_SIGNAL],
    }


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
