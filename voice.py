"""Speech in and out, both on Groq so one key covers the whole app."""
import os
import re
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_LIMIT = 200  # Orpheus rejects longer inputs


def _client():
    from groq import Groq

    return Groq()


def transcribe(audio: bytes, filename: str, keywords: list[str]) -> str:
    # Whisper's prompt biases spelling toward the candidate's own tools and project names.
    result = _client().audio.transcriptions.create(
        file=(filename, audio), model=STT_MODEL, language="en", prompt=", ".join(keywords)[:800],
    )
    return result.text.strip()


def chunk_text(text: str, limit: int = TTS_LIMIT) -> list[str]:
    """Split at sentence, then clause, then word boundaries so each piece fits the TTS limit."""
    chunks, current = [], ""
    for part in re.split(r"(?<=[.?!;:,])\s+", text.strip()):
        while len(part) > limit:
            cut = part.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        if current and len(current) + 1 + len(part) > limit:
            chunks.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def join_wavs(blobs: list[bytes]) -> bytes:
    """Concatenate same-format WAVs. Reads PCM after the data tag because streamed WAVs often carry bogus sizes."""
    with wave.open(BytesIO(blobs[0])) as first:
        channels, width, rate = first.getnchannels(), first.getsampwidth(), first.getframerate()
    out = BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        for blob in blobs:
            pcm = blob[blob.find(b"data") + 8:]
            writer.writeframes(pcm[: len(pcm) - len(pcm) % (channels * width)])
    return out.getvalue()


def synthesize(text: str, voice: str | None = None) -> bytes:
    client = _client()
    voice = voice or os.getenv("TTS_VOICE", "troy")

    def one(chunk: str) -> bytes:
        return client.audio.speech.create(model=TTS_MODEL, voice=voice, input=chunk, response_format="wav").read()

    chunks = chunk_text(text)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return join_wavs(list(pool.map(one, chunks)))
