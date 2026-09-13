"""
chatbot_engine.py — Shared matching engine for Study Buddy
===========================================================
Used by both app.py (terminal) and app_web.py (Streamlit).

Matching pipeline:
  1. Normalise – lowercase, expand abbreviations/synonyms, strip stop words.
  2. Stem      – reduce every token to a simple root (suffix stripping).
  3. Score     – count keyword overlap between query and each stored question.
  4. Fuzzy     – if exact-stem overlap is zero, fall back to fuzzy character
                 similarity so small typos still find the closest answer.
  5. AI        – if even fuzzy confidence is too low, call the Hugging Face
                 free Inference API (see ai_fallback.py) for a real AI answer.
  6. Static    – if the AI call fails (no internet, rate limit, etc.), show
                 a helpful static fallback message.
"""

import re
from ai_fallback import ask_ai

# ---------------------------------------------------------------------------
# Stop words — tokens that add no discriminating signal
# ---------------------------------------------------------------------------

STOP_WORDS: set[str] = {
    "what", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "the", "this", "these", "those", "that",
    "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "may", "might", "shall",
    "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "into", "from", "as", "how", "why", "when", "where", "who", "which",
    "i", "me", "my", "we", "our", "you", "your",
    "tell", "explain", "give", "please",
    "define", "definition", "meaning", "mean", "means",
    "describe", "difference", "between", "and", "or", "not",
    "some", "any", "just", "like", "make", "use", "used", "uses",
}

# ---------------------------------------------------------------------------
# Abbreviation / synonym map  (all keys lowercase, values are expansions)
# Expanded tokens are injected alongside the original so both forms match.
# ---------------------------------------------------------------------------

SYNONYMS: dict[str, list[str]] = {
    # Abbreviations → full form
    "ai":       ["artificial", "intelligence"],
    "ml":       ["machine", "learning"],
    "os":       ["operating", "system"],
    "oop":      ["object", "oriented", "programming"],
    "db":       ["database"],
    "api":      ["application", "programming", "interface"],
    "cpu":      ["central", "processing", "unit"],
    "ram":      ["random", "access", "memory"],
    "rom":      ["read", "only", "memory"],
    "git":      ["version", "control"],
    "sql":      ["database", "query"],
    "ui":       ["interface", "frontend"],
    "ux":       ["user", "experience", "interface"],
    # Casual phrasing → canonical tokens
    "cs":       ["computer", "science"],
    "prog":     ["programming"],
    "algo":     ["algorithm"],
    "func":     ["function"],
    "var":      ["variable"],
    "bool":     ["boolean"],
    "int":      ["integer"],
    "str":      ["string"],
    "arr":      ["array"],
    "struct":   ["structure"],
    "hw":       ["hardware"],
    "sw":       ["software"],
    "net":      ["network", "networking"],
    "cyber":    ["cybersecurity"],
    "cloud":    ["cloud", "computing"],
    "backend":  ["backend", "server"],
    "frontend": ["frontend", "client"],
    "debug":    ["debugging"],
    "test":     ["testing"],
    "compile":  ["compiler"],
    "interp":   ["interpreter"],
    "binary":   ["binary", "code"],
    "lang":     ["programming", "language"],
    "vc":       ["version", "control"],
    # Language name aliases — keep the language name token AND add "language"
    # so "what is python" and "what is python language" both hit the definition.
    "python":    ["python", "language"],
    "java":      ["java", "language"],
    "cpp":       ["cpp", "language"],
    "cplusplus": ["cpp", "language"],
    # "clang" is a proxy token injected when the query contains "what is c" or
    # "c language" — it expands to "programming" + "language" so it overlaps
    # with the C definition question ("What is the C programming language?").
    "clang":     ["programming", "language"],
}

# Language name tokens that must NOT be stripped as stop words.
_LANG_TOKENS: set[str] = {"python", "java", "cpp", "cplusplus", "clang"}

# ---------------------------------------------------------------------------
# Simple suffix-stripping stemmer
# (covers the most common English inflection patterns without NLTK)
# ---------------------------------------------------------------------------

_STEM_RULES: list[tuple[str, str]] = [
    # Order matters: longer suffixes first
    ("ations", ""),   ("ation", ""),
    ("ings",   ""),   ("ing",   ""),
    ("ments",  ""),   ("ment",  ""),
    ("nesses", ""),   ("ness",  ""),
    ("ities",  ""),   ("ity",   ""),
    ("iers",   ""),   ("ier",   ""),
    ("ical",   ""),   ("ically", ""),
    ("ively",  ""),   ("ive",   ""),
    ("ful",    ""),   ("fully", ""),
    ("less",   ""),
    ("ries",   "ry"), ("ries",  "r"),
    ("ves",    "f"),  ("ves",   "ve"),
    ("ies",    "y"),
    ("ses",    "s"),
    ("es",     ""),
    ("s",      ""),
    ("ed",     ""),
    ("er",     ""),
    ("ly",     ""),
]

_MIN_STEM_LENGTH = 4   # don't shorten tokens shorter than this


def stem(word: str) -> str:
    """Return a simple stemmed form of *word*."""
    if len(word) <= _MIN_STEM_LENGTH:
        return word
    for suffix, replacement in _STEM_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= _MIN_STEM_LENGTH:
            return word[: len(word) - len(suffix)] + replacement
    return word


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalise(text: str) -> list[str]:
    """
    Convert raw text into a list of meaningful stemmed tokens.

    Steps:
      1. Lowercase and tokenise (letters only).
      2. Expand abbreviations/synonyms (keep both original and expansion).
         Language names ("python", "java", "cpp") inject themselves + "language"
         so "what is python" reliably hits the definition entry.
      3. Remove stop words — but never remove language name tokens.
         The single-char token "c" is also kept when it appears adjacent to
         "language" in the query (handled by injecting "clang" as a proxy).
      4. Stem each token.
    """
    raw_tokens = re.findall(r"[a-z]+", text.lower())

    # Special handling: detect "c language" / "c programming" so the lone "c"
    # isn't silently dropped.  Insert a proxy token "clang" that survives
    # stop-word removal and matches the C definition question's keywords.
    lowered = text.lower()
    if re.search(r"\bc\s+(language|programming|lang)\b", lowered) or \
       re.search(r"\bwhat\s+is\s+c\b", lowered) or \
       re.search(r"\bthe\s+c\s+(language|programming)\b", lowered):
        raw_tokens.append("clang")   # proxy for "C language"

    expanded: list[str] = []
    for tok in raw_tokens:
        expanded.append(tok)
        if tok in SYNONYMS:
            expanded.extend(SYNONYMS[tok])

    # Keep language name tokens unconditionally; remove other stop words.
    filtered = [
        t for t in expanded
        if t not in STOP_WORDS or t in _LANG_TOKENS
    ]
    return [stem(t) for t in filtered]


# ---------------------------------------------------------------------------
# Fuzzy similarity helper  (Dice coefficient on character bigrams)
# ---------------------------------------------------------------------------

def _bigrams(s: str) -> list[str]:
    return [s[i: i + 2] for i in range(len(s) - 1)]


def _dice(a: str, b: str) -> float:
    """Return Dice similarity coefficient for two strings (0.0 – 1.0)."""
    if not a or not b:
        return 0.0
    ba, bb = set(_bigrams(a)), set(_bigrams(b))
    if not ba or not bb:
        return float(a == b)   # single-char: exact match only
    return 2 * len(ba & bb) / (len(ba) + len(bb))


def _token_similarity(query_tokens: list[str], question_tokens: list[str]) -> float:
    """
    Average best-match Dice score: for every query token find the most similar
    question token, then average those scores.
    """
    if not query_tokens or not question_tokens:
        return 0.0
    total = 0.0
    for qt in query_tokens:
        best = max(_dice(qt, qt2) for qt2 in question_tokens)
        total += best
    return total / len(query_tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Thresholds
_EXACT_MIN_OVERLAP    = 1      # minimum stems that must overlap for a hard match
_FUZZY_RETURN_THRESH  = 0.35   # fuzzy score below this → AI fallback attempted
_FUZZY_SUGGEST_THRESH = 0.20   # below this → AI fallback attempted (weaker match)

# Static message shown only when BOTH local matching AND the AI API fail
_STATIC_FALLBACK = (
    "I don't know that one yet, and I couldn't reach my AI backup right now. "
    "Try asking about AI, variables, loops, algorithms, functions, databases, "
    "machine learning, or networking — or check your internet connection!"
)


def find_best_answer(
    user_input: str,
    qa_pairs: list[dict],
) -> tuple[str, bool, bool]:
    """
    Match *user_input* against *qa_pairs* and return
    ``(response_text, is_exact, is_ai_generated)``.

    ``is_exact``        True  -> answered from the local Q&A dataset.
    ``is_ai_generated`` True  -> answered by the Hugging Face AI model.
    Both False          -> soft fuzzy suggestion or final static fallback.
    """
    query_tokens = normalise(user_input)

    # ── Phase 1: exact stem overlap (fast, offline) ────────────────────────
    best_overlap = 0
    best_pair: dict | None = None

    for pair in qa_pairs:
        q_tokens = normalise(pair["question"])
        overlap  = len(set(query_tokens) & set(q_tokens))
        if overlap > best_overlap:
            best_overlap = overlap
            best_pair    = pair

    if best_overlap >= _EXACT_MIN_OVERLAP:
        return best_pair["answer"], True, False

    # ── Phase 2: fuzzy / near-miss detection ──────────────────────────────
    best_fuzzy  = 0.0
    fuzzy_pair: dict | None = None

    for pair in qa_pairs:
        q_tokens = normalise(pair["question"])
        score    = _token_similarity(query_tokens, q_tokens)
        if score > best_fuzzy:
            best_fuzzy = score
            fuzzy_pair = pair

    # Good fuzzy match -> show the related topic from local data
    if best_fuzzy >= _FUZZY_RETURN_THRESH and fuzzy_pair:
        topic   = fuzzy_pair["question"]
        subject = re.sub(r"^what is (a |an |the )?", "", topic.lower()).rstrip("?")
        suggestion = (
            f"I don't have an exact answer for that, but here's something "
            f"related — **{subject}**:\n\n{fuzzy_pair['answer']}"
        )
        return suggestion, False, False

    # ── Phase 3: AI-powered fallback (Hugging Face Inference API) ─────────
    # Only reached when neither exact nor fuzzy matching found a good local answer.
    # ask_ai() returns None gracefully on any network / API error — never raises.
    ai_answer = ask_ai(user_input)

    if ai_answer:
        return ai_answer, False, True

    # ── Phase 4: static fallback — AI also unavailable ────────────────────
    # Show the closest topic hint if we at least have a weak fuzzy match
    if best_fuzzy >= _FUZZY_SUGGEST_THRESH and fuzzy_pair:
        topic   = fuzzy_pair["question"]
        subject = re.sub(r"^what is (a |an |the )?", "", topic.lower()).rstrip("?")
        return (
            f"Hmm, I'm not sure about that (and couldn't reach my AI backup). "
            f"The closest topic I know is **{subject}** — try asking about that, "
            f"or explore AI, variables, loops, algorithms, functions, or networking!",
            False,
            False,
        )

    return _STATIC_FALLBACK, False, False


# ---------------------------------------------------------------------------
# Data loading  (shared by both apps)
# ---------------------------------------------------------------------------

def load_qa_pairs(filepath: str) -> list[dict]:
    """
    Parse a Q&A text file into a list of {question, answer} dicts.

    Format:
        Q: question text
        A: answer text
        <blank line>
    """
    qa_pairs: list[dict] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        return qa_pairs

    blocks = re.split(r"\n\s*\n", raw.strip())
    for block in blocks:
        lines    = block.strip().splitlines()
        question = ""
        answer   = ""
        for line in lines:
            if line.startswith("Q:"):
                question = line[2:].strip()
            elif line.startswith("A:"):
                answer = line[2:].strip()
        if question and answer:
            qa_pairs.append({"question": question, "answer": answer})

    return qa_pairs


# ---------------------------------------------------------------------------
# Language registry
# ---------------------------------------------------------------------------
# Maps a language key to its dedicated data file.
# The general CS file is always used as a fallback when no language-specific
# match is found (see load_language_pairs).

LANGUAGES: dict[str, dict] = {
    "general": {
        "label":    "General CS",
        "emoji":    "💻",
        "file":     "data.txt",
        "fallback": None,           # no further fallback for general
    },
    "python": {
        "label":    "Python",
        "emoji":    "🐍",
        "file":     "data_python.txt",
        "fallback": "data.txt",
    },
    "java": {
        "label":    "Java",
        "emoji":    "☕",
        "file":     "data_java.txt",
        "fallback": "data.txt",
    },
    "c": {
        "label":    "C",
        "emoji":    "⚙️",
        "file":     "data_c.txt",
        "fallback": "data.txt",
    },
    "cpp": {
        "label":    "C++",
        "emoji":    "🔧",
        "file":     "data_cpp.txt",
        "fallback": "data.txt",
    },
}

# Ordered list for menu display (1-based for terminal, selectbox for web)
LANGUAGE_KEYS: list[str] = ["general", "python", "java", "c", "cpp"]


def load_language_pairs(language_key: str) -> tuple[list[dict], list[dict]]:
    """
    Load Q&A pairs for a given language key.

    Returns ``(lang_pairs, fallback_pairs)`` where:
      - ``lang_pairs``     is the language-specific dataset (may be empty if the
                           file is missing).
      - ``fallback_pairs`` is the general CS dataset used when no match is found
                           in the language-specific set.  For "general", this is
                           the same as lang_pairs and fallback_pairs is [].
    """
    info = LANGUAGES.get(language_key, LANGUAGES["general"])

    lang_pairs = load_qa_pairs(info["file"])

    if info["fallback"]:
        fallback_pairs = load_qa_pairs(info["fallback"])
    else:
        fallback_pairs = []   # "general" has no separate fallback

    return lang_pairs, fallback_pairs


def find_best_answer_for_language(
    user_input: str,
    lang_pairs: list[dict],
    fallback_pairs: list[dict],
) -> tuple[str, bool, bool]:
    """
    Two-stage language-aware lookup.

    1. Search *lang_pairs* first (the selected language's dataset).
    2. If no confident match is found, search *fallback_pairs* (General CS).
    3. If still no match, call the AI fallback then the static fallback.

    Returns the same ``(response_text, is_exact, is_ai_generated)`` tuple as
    ``find_best_answer()``.
    """
    # Search the language-specific dataset
    if lang_pairs:
        result = find_best_answer(user_input, lang_pairs)
        text, is_exact, is_ai = result
        if is_exact or is_ai:
            return result

    # No confident local match in the language set — try general CS fallback
    if fallback_pairs:
        result = find_best_answer(user_input, fallback_pairs)
        text, is_exact, is_ai = result
        if is_exact or is_ai:
            return result

    # Return whatever find_best_answer produced last (fuzzy suggestion or
    # static fallback) — or re-run on the combined set for best fuzzy pick
    combined = lang_pairs + fallback_pairs
    if combined:
        return find_best_answer(user_input, combined)

    return _STATIC_FALLBACK, False, False
