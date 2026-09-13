"""
ai_fallback.py — AI-powered fallback for Study Buddy
=====================================================
This is the "AI-powered" part of the project.

When the local keyword/fuzzy matching engine cannot find a good answer in
data.txt, this module calls the Hugging Face free Inference API to get a
real AI-generated answer to the user's question.

Model used: HuggingFaceH4/zephyr-7b-beta  (instruction-tuned, free tier)
  - Good at short factual Q&A, no login required for the free serverless API.
  - Falls back automatically to microsoft/phi-2 if the primary model is
    unavailable (e.g. cold-start timeout on the free tier).

Setup (optional):
  1. Create a free account at https://huggingface.co
  2. Generate a token at https://huggingface.co/settings/tokens  (read-only)
  3. Set the environment variable:
       Windows PowerShell:  $env:HF_TOKEN = "hf_..."
       bash/macOS:          export HF_TOKEN="hf_..."
  Without a token the API still works but at a lower rate limit.

Graceful degradation:
  - No internet   → returns None  (caller shows static fallback message)
  - Rate-limited  → returns None
  - Bad JSON      → returns None
  - Any exception → returns None
The app NEVER crashes due to an AI API failure.
"""

import os
import json
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Primary model: small, fast, good instruction following on free HF tier
_PRIMARY_MODEL   = "HuggingFaceH4/zephyr-7b-beta"
# Secondary model tried if primary is unavailable / cold-starting
_FALLBACK_MODEL  = "microsoft/phi-2"

_HF_API_BASE     = "https://api-inference.huggingface.co/models/"
_REQUEST_TIMEOUT = 15          # seconds — keeps the app responsive
_MAX_NEW_TOKENS  = 200         # short answers are fine for a study chatbot

# System prompt that frames the model as an educational CS assistant
_SYSTEM_PROMPT = (
    "You are Study Buddy, a friendly and concise Computer Science tutor. "
    "Answer the student's question clearly in 2-4 sentences. "
    "Focus on accuracy and simplicity. Do not add greetings or sign-offs."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_token() -> str:
    """Return the HF API token from the environment, or empty string."""
    return os.environ.get("HF_TOKEN", "").strip()


def _call_hf_api(model: str, user_question: str) -> str | None:
    """
    POST to the Hugging Face Inference API for *model*.
    Returns the generated text string, or None on any error.
    """
    url = (_HF_API_BASE + model).encode()

    # Build a simple chat-style prompt the model understands
    prompt = (
        f"<|system|>\n{_SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_question}\n"
        f"<|assistant|>\n"
    )

    payload = json.dumps({
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": _MAX_NEW_TOKENS,
            "temperature": 0.4,
            "return_full_text": False,  # return only the new tokens
        },
        "options": {
            "wait_for_model": True,     # wait up to 20 s for cold-start
            "use_cache": True,
        },
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    token = _get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        # Network error, timeout, or HTTP 4xx/5xx
        return None
    except json.JSONDecodeError:
        return None

    # HF returns a list: [{"generated_text": "..."}]
    try:
        text = body[0]["generated_text"].strip()
        return text if text else None
    except (KeyError, IndexError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask_ai(user_question: str) -> str | None:
    """
    ── AI-POWERED FALLBACK ──────────────────────────────────────────────────
    Send *user_question* to the Hugging Face free Inference API and return
    the model's answer as a plain string.

    Returns None if:
      - No internet connection
      - API rate limit hit
      - Model is unavailable / still loading after the timeout
      - Any other error

    The caller is responsible for showing a static fallback when None is
    returned — this function never raises.
    """
    # Try primary model first, then the fallback model
    for model in (_PRIMARY_MODEL, _FALLBACK_MODEL):
        result = _call_hf_api(model, user_question)
        if result:
            return result

    return None
