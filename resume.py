"""Resume in -> structured profile out.

Reads the resume text, follows the links inside it (GitHub, portfolio, project
pages) for extra context, then asks the LLM for a claim list the interviewer
can ground every question in.
"""
import ipaddress
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field
from pypdf import PdfReader

URL_RE = re.compile(r"(?:https?://|www\.|github\.com/)[^\s)>\]|,;\"']+", re.I)
SKIP_HOSTS = ("linkedin.com", "mailto", "leetcode.com", "codeforces.com")  # login walls / no signal
MAX_LINKS = 8


class Claim(BaseModel):
    id: str = Field(description="C1, C2, ... in order")
    text: str = Field(description="One concrete, checkable claim: what they did, with what, and the result")
    source: str = Field(description="'resume' or the URL it came from")
    competency: str = Field(description="Which competency from the list this claim evidences")
    importance: int = Field(ge=1, le=3, description="3 = central to the target role, 1 = minor")


class Profile(BaseModel):
    name: str
    headline: str = Field(description="One line: who this candidate is")
    competencies: list[str] = Field(description="4-6 competencies the target role needs that this resume can be probed on")
    claims: list[Claim] = Field(description="8-16 claims, most important first")
    keywords: list[str] = Field(description="Up to 40 proper nouns and technical terms (tools, projects, companies) for speech recognition")


def read_resume(data: bytes, filename: str) -> tuple[str, list[str]]:
    """Return (text, links). PDFs keep hyperlinks in annotations, not text, so read both."""
    links: list[str] = []
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        for page in reader.pages:
            for annot in page.get("/Annots") or []:
                action = annot.get_object().get("/A")
                uri = action.get_object().get("/URI") if action else None
                if uri:
                    links.append(str(uri))
    else:
        text = data.decode("utf-8", errors="ignore")
    links += URL_RE.findall(text)

    seen, clean = set(), []
    for link in links:
        link = link.rstrip(".")
        if not link.startswith("http"):
            link = "https://" + link
        key = link.lower().rstrip("/")
        if key not in seen and not any(h in key for h in SKIP_HOSTS):
            seen.add(key)
            clean.append(link)
    return text.strip(), clean[:MAX_LINKS]


def is_public_url(url: str) -> bool:
    """Links come from an uploaded file, so refuse anything pointing at private/internal hosts."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        return False
    return all(ipaddress.ip_address(info[4][0]).is_global for info in infos)


def _github(client: httpx.Client, path: list[str]) -> str:
    if len(path) >= 2:  # repo
        repo = client.get(f"https://api.github.com/repos/{path[0]}/{path[1]}").json()
        readme = client.get(
            f"https://api.github.com/repos/{path[0]}/{path[1]}/readme",
            headers={"Accept": "application/vnd.github.raw"},
        )
        return (
            f"Repo {repo.get('full_name')}: {repo.get('description')} | lang={repo.get('language')} "
            f"stars={repo.get('stargazers_count')} topics={repo.get('topics')}\nREADME:\n"
            + (readme.text[:3000] if readme.status_code == 200 else "")
        )
    repos = client.get(f"https://api.github.com/users/{path[0]}/repos", params={"sort": "pushed", "per_page": 12}).json()
    if not isinstance(repos, list):
        return ""
    return "GitHub repos:\n" + "\n".join(
        f"- {r['name']}: {r.get('description') or ''} (lang={r.get('language')}, stars={r['stargazers_count']})"
        for r in repos if not r.get("fork")
    )


def _fetch(url: str) -> tuple[str, str]:
    if not is_public_url(url):
        return url, ""
    try:
        with httpx.Client(timeout=6, follow_redirects=True, headers={"User-Agent": "PrepAI/1.0"}) as client:
            parsed = urlparse(url)
            path = [p for p in parsed.path.split("/") if p]
            if parsed.hostname.endswith("github.com") and path:
                return url, _github(client, path)
            resp = client.get(url)
            if "html" not in resp.headers.get("content-type", ""):
                return url, ""
            text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", resp.text, flags=re.S | re.I)
            text = re.sub(r"<[^>]+>", " ", text)
            return url, re.sub(r"\s+", " ", text)[:2500]
    except (httpx.HTTPError, ValueError):
        return url, ""


def fetch_resources(links: list[str]) -> dict[str, str]:
    """Fetch every link in parallel; drop the ones that gave nothing."""
    with ThreadPoolExecutor(max_workers=MAX_LINKS) as pool:
        return {url: text for url, text in pool.map(_fetch, links) if text.strip()}


PROFILE_PROMPT = """You prepare a technical interviewer for a {role} interview.

Extract the candidate profile from the resume and the linked resources below.
Rules:
- Claims must be specific and checkable ("Built X with Y, cut Z by 40%"), never generic ("team player").
- Only use what is written. Never invent numbers, tools or outcomes.
- Prefer claims an interviewer could dig into for 2-3 follow-up questions.
- Linked resources may add detail to a resume claim; set source to that URL.

RESUME:
{resume}

LINKED RESOURCES:
{resources}"""


def extract_profile(structured, resume_text: str, resources: dict[str, str], role: str) -> Profile:
    """`structured(schema)` returns a runnable that yields that schema (graph.structured)."""
    blob = "\n\n".join(f"[{url}]\n{text}" for url, text in resources.items()) or "(none)"
    prompt = PROFILE_PROMPT.format(role=role, resume=resume_text[:12000], resources=blob[:12000])
    return structured(Profile).invoke(prompt)
