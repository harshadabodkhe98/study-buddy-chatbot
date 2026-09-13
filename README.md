# 📚 Study Buddy — AI Chatbot

A hackathon project: a **hybrid AI chatbot** that answers Computer Science
and programming language questions. Choose a language (Python, Java, C, or C++),
ask questions in plain English, and get answers from a curated dataset or a
real AI model. Available as a terminal app and a polished Streamlit web interface.

---

## Language selection

| # | Language | Data file | Q&A pairs |
|---|---|---|---|
| 1 | 💻 General CS | `data.txt` | 33 |
| 2 | 🐍 Python | `data_python.txt` | 17 |
| 3 | ☕ Java | `data_java.txt` | 17 |
| 4 | ⚙️ C | `data_c.txt` | 16 |
| 5 | 🔧 C++ | `data_cpp.txt` | 17 |

- **Terminal app**: prompted at startup with a numbered menu (1–5); type `switch` at any time to change.
- **Web app**: a selectbox at the top of the page; switching automatically clears and resets the chat.

---

## How the hybrid matching pipeline works

```
User question
      │
      ▼
┌─────────────────────────────────┐
│  Phase 1 — Language KB          │  ← instant, offline
│  Keyword + stem match in the    │
│  selected language's data file  │
└────────────┬────────────────────┘
             │ no confident match?
             ▼
┌─────────────────────────────────┐
│  Phase 2 — General CS KB        │  ← instant, offline
│  Same matching in data.txt      │
│  (fallback for all languages)   │
└────────────┬────────────────────┘
             │ still no match?
             ▼
┌─────────────────────────────────┐   ← AI-POWERED PART
│  Phase 3 — Hugging Face AI      │
│  Free Inference API             │
│  Model: HuggingFaceH4/zephyr-7b │
└────────────┬────────────────────┘
             │ API unavailable / rate-limited?
             ▼
┌─────────────────────────────────┐
│  Phase 4 — Static fallback      │  ← always friendly, never crashes
└─────────────────────────────────┘
```

**Why this design?**
- Phases 1 & 2 are instant and fully offline — they cover almost all typical questions.
- Phase 3 handles niche topics, follow-up questions, or creative phrasing with a real LLM.
- Graceful degradation: the app works without internet or an API key — it just skips Phase 3.

---

## Project structure

```
Hackathon/
├── app.py              # Terminal chatbot (with language menu)
├── app_web.py          # Streamlit web interface (with language selectbox)
├── chatbot_engine.py   # Shared matching engine + language registry
├── ai_fallback.py      # Hugging Face Inference API integration
├── data.txt            # General CS — 33 Q&A pairs
├── data_python.txt     # Python — 17 Q&A pairs
├── data_java.txt       # Java   — 17 Q&A pairs
├── data_c.txt          # C      — 16 Q&A pairs
├── data_cpp.txt        # C++    — 17 Q&A pairs
├── requirements.txt
└── README.md
```

---

## Setup

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

> `streamlit` and `requests` are the only dependencies.
> `app.py` and `chatbot_engine.py` / `ai_fallback.py` use only the Python
> standard library for HTTP (`urllib`) so no extra install is needed for the
> terminal version — `requests` is listed for completeness.

### 2 · (Optional) Set a Hugging Face API token

The AI fallback works **without** a token on the free public rate limit.
To increase that limit:

1. Create a free account at <https://huggingface.co>
2. Generate a **read-only** token at <https://huggingface.co/settings/tokens>
3. Set the environment variable before running the app:

```powershell
# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"
```

```bash
# macOS / Linux
export HF_TOKEN="hf_your_token_here"
```

---

## How to run

### Terminal version

```bash
python app.py
```

At startup you are prompted to pick a topic:

```
  Select a topic / language to study:

  1. 💻  General CS
  2. 🐍  Python
  3. ☕  Java
  4. ⚙️  C
  5. 🔧  C++

  Enter 1-5: 2
```

Each response is labelled with its source:

```
You: How do I print in Python?
Bot [Python KB]: Use the built-in print() function...

You: What is quantum entanglement?
Bot [AI-powered]: Quantum entanglement is a phenomenon...

You: switch
  (language menu appears again)
```

Type `exit`, `quit`, or `bye` to quit. Type `switch` to change language mid-session.

---

### Web version (Streamlit)

```bash
streamlit run app_web.py
```

Opens automatically at `http://localhost:8501`.

- A **language selectbox** at the top lets you switch topics instantly
- Switching language **automatically clears** the chat and shows a new welcome message
- AI-generated answers are clearly badged **"🤖 Answered by Hugging Face AI model"**
- The sidebar shows loaded dataset sizes, HF token status, and sample questions per language
- A **Clear chat** button resets the conversation without changing language

---

## Technologies used

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Regular expressions (`re`)** | Tokenising, parsing, stemming |
| **Custom suffix stemmer** | Plural/inflection normalisation without NLTK |
| **Dice coefficient** | Fuzzy bigram similarity for typo tolerance |
| **Hugging Face Inference API** | Free AI-powered fallback (no GPU needed) |
| **`urllib` (stdlib)** | HTTP requests to the HF API — no extra install |
| **Streamlit** | Browser-based chat UI |
| **Session state** | Persisting conversation history in the web app |

---

## Matching features at a glance

| Feature | Example |
|---|---|
| Abbreviation expansion | `AI` → artificial intelligence |
| Plural/singular | `loops` → `loop`, `variables` → `variable` |
| Casual phrasing | *"explain X"*, *"tell me about X"*, *"define X"* |
| Typo tolerance | `veriable` still matches **variable** |
| AI fallback | Questions outside data.txt answered by Zephyr-7b |
| Graceful degradation | Works fully offline (local KB only) |

---

## Extending the knowledge base

Add more pairs to `data.txt` (blank line between each):

```
Q: Your question here?
A: Your answer here.
```

No code changes needed — the app picks up new entries on the next run.

---

*Made for a hackathon — happy hacking!*
