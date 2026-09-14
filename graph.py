"""The interview loop as a LangGraph state machine.

    ask -> listen (interrupt: wait for the candidate) -> grade -> decide -+-> ask
                                                                         +-> report -> END

Every answer is graded against a rubric and appended to `turns`. `decide` is plain
Python over those grades: it picks the next move (dig deeper, go harder, challenge a
contradiction, change topic, wrap up). The final report reads the same `turns`, so the
evaluation is accumulated answer by answer rather than guessed from the transcript.
"""
import os
from statistics import mean
from typing import Literal, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

END_SIGNAL = "__end_interview__"
FOLLOW_UPS_PER_CLAIM = 2

_llm = None


def llm():
    """One chat model for everything; swap providers with LLM_MODEL (any init_chat_model string)."""
    global _llm
    if _llm is None:
        model = os.getenv("LLM_MODEL", "groq:openai/gpt-oss-120b")
        extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
        _llm = init_chat_model(model, temperature=0.4, **extra)
    return _llm


# ---------------------------------------------------------------- schemas

class Grade(BaseModel):
    specificity: int = Field(ge=1, le=5, description="1 vague generalities ... 5 concrete names, numbers, decisions")
    ownership: int = Field(ge=1, le=5, description="1 only 'we'/the team ... 5 clearly what THEY did and decided")
    depth: int = Field(ge=1, le=5, description="1 surface buzzwords ... 5 explains how/why, tradeoffs, failure modes")
    impact: int = Field(ge=1, le=5, description="1 no outcome ... 5 measurable result tied to their work")
    clarity: int = Field(ge=1, le=5, description="1 rambling/unstructured ... 5 crisp and easy to follow")
    contradicts_resume: bool = Field(description="Answer conflicts with the resume claim (dates, scope, role, numbers)")
    contradiction: str = Field(description="What conflicts, empty if nothing")
    off_topic: bool = Field(description="Answer does not address the question at all")
    evidence: str = Field(description="Shortest quote from the answer that best supports the scores")
    note: str = Field(description="One line assessor note: what was strong or missing")


class CompetencyScore(BaseModel):
    name: str
    score: int = Field(ge=1, le=10)
    evidence: str = Field(description="Quote or paraphrase from the interview backing this score")


class Rewrite(BaseModel):
    question: str
    what_was_missing: str
    stronger_answer: str = Field(description="How the candidate could have answered, using only facts they stated or that are on their resume")


class Report(BaseModel):
    verdict: Literal["Strong hire", "Hire", "Lean hire", "Lean no hire", "No hire"]
    summary: str = Field(description="3-4 sentences, direct, no flattery")
    strengths: list[str]
    gaps: list[str]
    competencies: list[CompetencyScore]
    rewrites: list[Rewrite] = Field(description="The 3 weakest answers, weakest first")


class Turn(TypedDict):
    claim_id: str
    move: str
    question: str
    answer: str
    grade: dict


class State(TypedDict, total=False):
    role: str
    profile: dict
    budget: int
    turns: list[Turn]
    claim_id: str
    move: str
    depth: int
    difficulty: int
    question: str
    answer: str
    report: dict
    score: float


# ---------------------------------------------------------------- pure logic

def answer_quality(grade: dict) -> float:
    """0..1 from the five rubric dimensions."""
    return (mean(grade[k] for k in ("specificity", "ownership", "depth", "impact", "clarity")) - 1) / 4


def overall_score(turns: list[Turn]) -> float:
    """Rubric average on a 10 point scale. Deterministic, so the headline number can't drift with the LLM."""
    graded = [answer_quality(t["grade"]) for t in turns if t["grade"]]
    return round(10 * mean(graded), 1) if graded else 0.0


def pick_claim(profile: dict, turns: list[Turn]) -> str | None:
    """Next unasked claim, favouring competencies with the least evidence so far, then importance."""
    asked = {t["claim_id"] for t in turns}
    by_id = {c["id"]: c for c in profile["claims"]}
    evidence: dict[str, int] = {}
    for t in turns:
        comp = by_id.get(t["claim_id"], {}).get("competency")
        evidence[comp] = evidence.get(comp, 0) + 1
    fresh = [c for c in profile["claims"] if c["id"] not in asked]
    if not fresh:
        return None
    return min(fresh, key=lambda c: (evidence.get(c["competency"], 0), -c["importance"]))["id"]


def next_move(state: State) -> dict:
    """Decide what the next question should do, based on the answer just graded."""
    turns = state["turns"]
    last = turns[-1]
    left = state["budget"] - len(turns)
    depth, difficulty = state.get("depth", 0), state.get("difficulty", 2)

    if last["answer"] == END_SIGNAL or left <= 0:
        return {"move": "wrap"}

    grade = last["grade"]
    q = answer_quality(grade)
    same_claim = {"claim_id": last["claim_id"], "depth": depth + 1}
    asked = {t["claim_id"] for t in turns}
    must_cover = sum(1 for c in state["profile"]["claims"] if c["importance"] == 3 and c["id"] not in asked)
    can_stay = depth < FOLLOW_UPS_PER_CLAIM and must_cover < left  # breadth beats depth when time runs short

    if grade["contradicts_resume"] and last["move"] != "verify" and can_stay:
        return {"move": "verify", "difficulty": difficulty, **same_claim}
    if grade["off_topic"] and last["move"] != "redirect" and can_stay:
        return {"move": "redirect", "difficulty": difficulty, **same_claim}
    if can_stay and q >= 0.75:
        return {"move": "escalate", "difficulty": min(5, difficulty + 1), **same_claim}
    if can_stay and (q < 0.45 or depth == 0):
        return {"move": "probe_deeper", "difficulty": difficulty, **same_claim}

    claim = pick_claim(state["profile"], turns)
    if claim is None:
        return {"move": "wrap"}
    difficulty = max(1, difficulty - 1) if q < 0.45 else difficulty
    return {"move": "switch_topic", "claim_id": claim, "depth": 0, "difficulty": difficulty}


# ---------------------------------------------------------------- nodes

MOVES = {
    "open": "Greet {name} in one short sentence, then ask your first question about the claim.",
    "probe_deeper": "Their last answer was thin. Ask for the concrete specifics of what THEY personally did, decided or measured. Point at the vague part.",
    "escalate": "Their last answer was strong. Go one level harder on the same work: tradeoffs, why not an alternative, what broke, how it scales.",
    "verify": "Their answer conflicts with the resume ({contradiction}). Point out the mismatch neutrally and ask them to reconcile it.",
    "redirect": "They didn't answer the question. Briefly steer them back to it, rephrased more simply.",
    "switch_topic": "Move to a new topic with a short transition, then ask about the claim.",
}

ASK_PROMPT = """You are a sharp, friendly interviewer running a live VOICE interview for a {role} role.
Candidate: {name} - {headline}
Difficulty: {difficulty}/5

Current resume claim [{claim_id}] ({competency}): {claim}

Recent conversation:
{history}

Your task for this turn: {task}

Rules: this will be spoken aloud. One question only, at most 2 sentences and 40 words.
No lists, no markdown, no emojis. Acknowledge the previous answer in at most 4 words, or not at all.
Never praise excessively and never reveal how they are scoring. Output only what you say."""


def ask(state: State) -> dict:
    profile, turns = state["profile"], state.get("turns", [])
    move = state.get("move", "open")
    claim_id = state.get("claim_id") or pick_claim(profile, turns)
    claim = next(c for c in profile["claims"] if c["id"] == claim_id)
    history = "\n".join(f"Q: {t['question']}\nA: {t['answer']}" for t in turns[-3:]) or "(interview just started)"
    contradiction = turns[-1]["grade"].get("contradiction", "") if turns else ""
    prompt = ASK_PROMPT.format(
        role=state["role"], name=profile["name"], headline=profile["headline"],
        difficulty=state.get("difficulty", 2), claim_id=claim_id, competency=claim["competency"],
        claim=claim["text"], history=history,
        task=MOVES[move].format(name=profile["name"].split()[0], contradiction=contradiction),
    )
    return {"question": llm().invoke(prompt).content.strip(), "claim_id": claim_id, "move": move}


def listen(state: State) -> dict:
    # Nothing may run before interrupt(): LangGraph re-executes this node on resume.
    return {"answer": interrupt({"question": state["question"], "index": len(state.get("turns", [])) + 1})}


GRADE_PROMPT = """Grade one answer from a {role} interview. Be strict and calibrated:
a 3 is an ordinary answer, 5 is rare. Speech-to-text may garble words; don't penalise that.

Resume claim being discussed: {claim}
Question: {question}
Answer: {answer}"""


def grade(state: State) -> dict:
    turns = list(state.get("turns", []))
    answer = state["answer"]
    result = {}
    if answer != END_SIGNAL:
        claim = next(c for c in state["profile"]["claims"] if c["id"] == state["claim_id"])
        prompt = GRADE_PROMPT.format(role=state["role"], claim=claim["text"], question=state["question"], answer=answer)
        result = llm().with_structured_output(Grade).invoke(prompt).model_dump()
    turns.append(Turn(claim_id=state["claim_id"], move=state["move"], question=state["question"], answer=answer, grade=result))
    return {"turns": turns}


REPORT_PROMPT = """Write the evaluation for a {role} interview.
Candidate: {name} - {headline}
Competencies to score: {competencies}

Each turn below has the question, the answer, and a per-answer rubric grade (1-5).
Base every judgement on what was actually said. Quote them.

{transcript}"""


def report(state: State) -> dict:
    turns = [t for t in state["turns"] if t["grade"]]
    if not turns:
        return {"report": {}, "score": 0.0}
    profile = state["profile"]
    transcript = "\n\n".join(
        f"Q{i}: {t['question']}\nA{i}: {t['answer']}\nGrade: {t['grade']}" for i, t in enumerate(turns, 1)
    )
    prompt = REPORT_PROMPT.format(
        role=state["role"], name=profile["name"], headline=profile["headline"],
        competencies=", ".join(profile["competencies"]), transcript=transcript,
    )
    result = llm().with_structured_output(Report).invoke(prompt)
    return {"report": result.model_dump(), "score": overall_score(turns)}


def build_graph(checkpointer):
    g = StateGraph(State)
    g.add_node(ask)
    g.add_node(listen)
    g.add_node(grade)
    g.add_node("decide", next_move)
    g.add_node(report)
    g.add_edge(START, "ask")
    g.add_edge("ask", "listen")
    g.add_edge("listen", "grade")
    g.add_edge("grade", "decide")
    g.add_conditional_edges("decide", lambda s: "report" if s["move"] == "wrap" else "ask", ["ask", "report"])
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)
