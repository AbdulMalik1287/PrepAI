# PrepAI

Voice mock interviewer built from your resume. Upload a resume, pick a role, and talk.
Every answer is graded as you go, and that grade decides what the next question does:
dig deeper, go harder, challenge a contradiction with your resume, or move on.
The final report is built from the same per-answer grades.

## Run

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements.txt
cp .env.example .env              # add GROQ_API_KEY
uvicorn server:app --reload
```

Open http://127.0.0.1:8000 in Chrome or Edge. Allow the microphone.

## How it works

```
resume.pdf ─> text + links inside it ─> fetch GitHub / portfolio pages ─> LLM: claims + competencies
                                                                                   │
        ┌──────────────────────────── LangGraph loop ──────────────────────────────┘
        ▼
       ask ──> listen (interrupt, waits for your answer) ──> grade ──> decide ──┬──> ask
                                                                                └──> report
```

| Piece | Choice |
|---|---|
| Orchestration | LangGraph `StateGraph` with `interrupt()` per answer, SQLite checkpointer |
| LLM | `init_chat_model`, default `groq:openai/gpt-oss-120b` (set `LLM_MODEL` for any provider) |
| Speech to text | Groq `whisper-large-v3-turbo`, biased with names and tools from your resume |
| Text to speech | Browser `speechSynthesis`, with word-by-word highlighting |
| Turn detection | Browser mic level: answer ends after 1.6 s of silence, or tap the mic |

**The adaptive part** is `next_move()` in `graph.py`. It's plain Python over the rubric
(specificity, ownership, depth, impact, clarity, each 1-5), not an LLM call, so it's testable:

| After your answer | Next question |
|---|---|
| Contradicts the resume | `verify`: asks you to reconcile it (once) |
| Off topic | `redirect` (once) |
| Strong (≥ 0.75) | `escalate`: harder follow-up on the same work, difficulty +1 |
| Weak, or first answer on a topic | `probe_deeper`: asks what *you* did, specifically |
| 2 follow-ups done, or too few questions left for the key claims | `switch_topic` to the least-covered competency |
| Out of questions, or you end it | `report` |

The question writer never sees your scores, only the move. The headline score is the
rubric average computed in Python. The LLM writes the verdict, competency notes and
"answer it better" rewrites.

As in the reference video's sideband design, prompts, transcript and grades stay on the server.
The browser only sends audio, so a client can't rewrite its own interview.

## Files

| File | What |
|---|---|
| `resume.py` | PDF/TXT parsing, link extraction (PDF annotations too), resource fetching with a private-IP guard, profile extraction |
| `graph.py` | State, rubric schemas, router, prompts, graph |
| `server.py` | FastAPI: `POST /api/interviews`, `POST /api/interviews/{id}/answer`, `POST /api/interviews/{id}/end`, `GET /api/interviews/{id}` |
| `static/` | Single-page UI: setup, live interview, report |
| `test_graph.py` | `python test_graph.py`: router rules plus a full interview through the real graph with a fake LLM |

## Not in v1

LinkedIn scraping (login wall), barge-in while the AI is talking (you can tap to skip),
streamed TTS voices (ElevenLabs etc.), audio recording, accounts.
