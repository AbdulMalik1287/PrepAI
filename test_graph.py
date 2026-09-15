"""Self-check: adaptive routing + a full interview through the real graph with a fake LLM.

Run: python test_graph.py
"""
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import graph
from graph import END_SIGNAL, Grade, Report, build_graph, next_move, overall_score, pick_claim
from resume import is_public_url, read_resume

PROFILE = {
    "name": "Asha Rao",
    "headline": "ML engineer",
    "competencies": ["ML systems", "Backend", "Ownership"],
    "claims": [
        {"id": "C1", "text": "Built a GNN EEG classifier, AUC 0.88", "source": "resume", "competency": "ML systems", "importance": 3},
        {"id": "C2", "text": "Shipped FastAPI recon engine, 98.6% recall", "source": "resume", "competency": "Backend", "importance": 3},
        {"id": "C3", "text": "Led a 4 person hackathon team", "source": "resume", "competency": "Ownership", "importance": 2},
        {"id": "C4", "text": "Wrote a GA scheduler", "source": "resume", "competency": "ML systems", "importance": 1},
    ],
}


def g(score, contradicts=False, off_topic=False):
    return dict(specificity=score, ownership=score, depth=score, impact=score, clarity=score,
                contradicts_resume=contradicts, contradiction="said 2 weeks, resume says 6 months" if contradicts else "",
                off_topic=off_topic, evidence="", note="")


def state(*turns, depth=0, budget=8, difficulty=2):
    return {"profile": PROFILE, "budget": budget, "depth": depth, "difficulty": difficulty,
            "turns": [{"claim_id": c, "move": m, "question": "q", "answer": "a", "grade": gr} for c, m, gr in turns]}


def test_router():
    # weak answer -> dig into the same claim
    assert next_move(state(("C1", "open", g(2)))) =={"move": "probe_deeper", "claim_id": "C1", "depth": 1, "difficulty": 2}
    # strong answer -> harder question, same claim
    assert next_move(state(("C1", "open", g(5))))["move"] == "escalate"
    assert next_move(state(("C1", "open", g(5))))["difficulty"] == 3
    # contradiction beats everything, but only challenged once
    assert next_move(state(("C1", "open", g(5, contradicts=True))))["move"] == "verify"
    assert next_move(state(("C1", "verify", g(5, contradicts=True)), depth=1))["move"] == "escalate"
    # off topic -> redirect once
    assert next_move(state(("C1", "open", g(1, off_topic=True))))["move"] == "redirect"
    # follow-up cap -> new claim from the least-evidenced competency (C2 Backend, not C4 ML systems)
    m = next_move(state(("C1", "escalate", g(5)), depth=2))
    assert (m["move"], m["claim_id"], m["depth"]) == ("switch_topic", "C2", 0), m
    # middling answer after a follow-up -> move on
    assert next_move(state(("C1", "probe_deeper", g(3)), depth=1))["move"] == "switch_topic"
    # weak answer on a forced switch lowers difficulty
    assert next_move(state(("C1", "probe_deeper", g(1)), depth=2, difficulty=3))["difficulty"] == 2
    # breadth guard: 1 question left and critical C2 still unasked -> switch instead of escalating
    assert next_move(state(("C1", "open", g(5)), budget=2))["move"] == "switch_topic"
    # budget exhausted or candidate ended
    assert next_move(state(("C1", "open", g(5)), budget=1)) == {"move": "wrap"}
    assert next_move(state(("C1", "open", {}), budget=8) | {"turns": [{"claim_id": "C1", "move": "open", "question": "q", "answer": END_SIGNAL, "grade": {}}]}) == {"move": "wrap"}


def test_scoring():
    assert pick_claim(PROFILE, []) == "C1"
    assert overall_score([{"grade": g(5)}, {"grade": g(1)}, {"grade": {}}]) == 5.0


class FakeLLM:
    """Grades answers by length so the script can steer the router; records the moves it was asked for."""

    def __init__(self):
        self.tasks = []

    def invoke(self, prompt):
        self.tasks.append(prompt.split("Your task for this turn: ")[1].split("\n")[0])
        return AIMessage(content=f"Question {len(self.tasks)}?")

    def with_structured_output(self, schema, **kwargs):
        assert kwargs == {"method": "json_schema", "strict": True}, kwargs
        class Runner:
            def invoke(self, prompt):
                if schema is Grade:
                    answer = prompt.split("Answer: ")[1]
                    return Grade(**g(5 if len(answer) > 40 else 2))
                return Report(verdict="Hire", summary="s", strengths=["a"], gaps=["b"], competencies=[], rewrites=[])
        return Runner()


def test_full_interview():
    fake = FakeLLM()
    graph._llm = fake
    app = build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    out = app.invoke({"role": "ML Engineer", "profile": PROFILE, "budget": 4, "turns": []}, cfg)
    assert out["__interrupt__"][0].value == {"question": "Question 1?", "index": 1}

    answers = ["idk", "I personally wrote the message passing layer and tuned it to 0.88 AUC", "much longer answer about tradeoffs and scaling", "short"]
    for i, answer in enumerate(answers):
        out = app.invoke(Command(resume=answer), cfg)
        if i < len(answers) - 1:
            assert "__interrupt__" in out

    moves = [t["move"] for t in out["turns"]]
    assert moves == ["open", "probe_deeper", "escalate", "switch_topic"], moves
    assert out["report"]["verdict"] == "Hire"
    assert out["score"] == overall_score(out["turns"])
    assert "__interrupt__" not in out

    # ending early still produces a report from what was answered
    cfg2 = {"configurable": {"thread_id": "t2"}}
    app.invoke({"role": "ML Engineer", "profile": PROFILE, "budget": 8, "turns": []}, cfg2)
    app.invoke(Command(resume="a long enough answer to count as a strong one here"), cfg2)
    out = app.invoke(Command(resume=END_SIGNAL), cfg2)
    assert len(out["turns"]) == 2 and out["report"] and out["score"] == 10.0


def test_resume_links():
    text, links = read_resume(b"Asha - github.com/asha/gnn-eeg, https://asha.dev. linkedin.com/in/asha", "r.txt")
    assert links == ["https://github.com/asha/gnn-eeg", "https://asha.dev"], links
    assert not is_public_url("http://127.0.0.1:8000/admin")
    assert not is_public_url("file:///etc/passwd")


def test_voice_helpers():
    import wave
    from io import BytesIO

    from voice import chunk_text, join_wavs

    question = "Walk me through the graph you built for the EEG classifier, and why GCN layers over attention. " * 3
    chunks = chunk_text(question)
    assert all(len(c) <= 200 for c in chunks) and " ".join(chunks) == question.strip(), chunks
    assert chunk_text("x" * 450) == ["x" * 200, "x" * 200, "x" * 50]
    assert chunk_text("Short one?") == ["Short one?"]

    def wav(frames: int, bogus_size: bool = False) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000)
            w.writeframes(b"\x01\x00" * frames)
        data = buf.getvalue()
        return data[:40] + b"\xff\xff\xff\xff" + data[44:] if bogus_size else data  # streamed-WAV style header

    with wave.open(BytesIO(join_wavs([wav(100), wav(50, bogus_size=True)]))) as joined:
        assert (joined.getnframes(), joined.getframerate()) == (150, 24000)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
