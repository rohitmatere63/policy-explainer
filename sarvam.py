"""Minimal Sarvam AI client: chat, translate, text-to-speech."""
import json
import os
import re
from functools import lru_cache

import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()  # must run before the os.getenv() lookups below
except ImportError:
    pass

CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b")
TRANSLATE_MODEL = os.getenv("SARVAM_TRANSLATE_MODEL", "mayura:v1")
TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "shubh")


def _key() -> str:
    key = os.getenv("SARVAM_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("SARVAM_API_KEY", "")
        except Exception:
            pass
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set (.env locally, Secrets on Streamlit Cloud)")
    return key


def _post(path: str, payload: dict) -> dict:
    r = httpx.post(f"https://api.sarvam.ai{path}", json=payload, timeout=90,
                   headers={"api-subscription-key": _key()})
    if r.status_code >= 400:  # show Sarvam's own message instead of a bare "400"
        raise RuntimeError(f"Sarvam {path} -> {r.status_code}: {r.text[:500]}")
    return r.json()

def chat_json(system: str, user: str, temperature: float = 0.1) -> dict | None:
    for _ in range(2):  # retry once if the JSON is malformed
        out = _post("/v1/chat/completions", {
            "model": CHAT_MODEL, "temperature": temperature, "max_tokens": 3000,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        })["choices"][0]["message"]["content"] or ""
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S)
        m = re.search(r"\{.*\}", out, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _split(text: str, limit: int = 900) -> list[str]:
    parts, cur = [], ""
    for s in re.split(r"(?<=[.!?।])\s+", text.strip()):
        if cur and len(cur) + len(s) + 1 > limit:
            parts.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    return parts + [cur] if cur else [text]


@lru_cache(maxsize=1024)
def translate(text: str, target: str, source: str = "en-IN") -> str:
    if not text.strip() or target == source:
        return text
    return " ".join(_post("/translate", {
        "input": p, "source_language_code": source, "target_language_code": target,
        "model": TRANSLATE_MODEL})["translated_text"] for p in _split(text))


def text_to_speech(text: str, lang: str) -> str | None:
    """Returns base64 WAV (first 500 chars)."""
    audios = _post("/text-to-speech", {
        "inputs": [text[:500]], "target_language_code": lang,
        "speaker": TTS_SPEAKER, "model": TTS_MODEL}).get("audios", [])
    return audios[0] if audios else None
