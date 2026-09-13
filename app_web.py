"""
Study Buddy — Streamlit Web Interface
======================================
Browser-based chat UI powered by chatbot_engine.py.

The language/topic dropdown sits at the top of the page.
The standard st.chat_input bar at the bottom handles message submission.

Matching pipeline:
  1. LOCAL (language) — keyword + stem matching in the chosen language's data
  2. LOCAL (general)  — same matching in data.txt when tier-1 finds nothing
  3. AI               — Hugging Face free Inference API as a last resort
  4. STATIC           — friendly message when all other tiers fail

Run:  streamlit run app_web.py
"""

import os
import streamlit as st
from chatbot_engine import (
    LANGUAGES,
    LANGUAGE_KEYS,
    load_language_pairs,
    find_best_answer_for_language,
)


# ---------------------------------------------------------------------------
# Data loading (cached per language key for the session lifetime)
# ---------------------------------------------------------------------------

@st.cache_data
def get_language_pairs(language_key: str) -> tuple[list[dict], list[dict]]:
    """Load and cache Q&A pairs for the given language key."""
    return load_language_pairs(language_key)


# ---------------------------------------------------------------------------
# Response handler
# ---------------------------------------------------------------------------

def _handle_user_input(
    user_input: str,
    lang_pairs: list[dict],
    fallback_pairs: list[dict],
    lang_info: dict,
) -> None:
    """Display the user message, generate the bot response, store both."""
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input,
        "is_ai": False,
    })

    with st.spinner("Thinking…"):
        response, is_exact, is_ai = find_best_answer_for_language(
            user_input, lang_pairs, fallback_pairs
        )

    with st.chat_message("assistant"):
        st.markdown(response)
        if is_ai:
            st.caption("🤖 Answered by Hugging Face AI model")
        elif not is_exact:
            st.caption("💡 Tip: rephrase your question for a more precise answer.")

    st.session_state["messages"].append({
        "role": "assistant",
        "content": response,
        "is_ai": is_ai,
    })


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Study Buddy - AI Chatbot",
        page_icon="📚",
        layout="centered",
    )

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("📚 Study Buddy — AI Chatbot")
    st.caption(
        "Ask questions about Computer Science or a specific programming language. "
        "Powered by local keyword matching + Hugging Face AI fallback."
    )

    # ── Language selector (top of page) ─────────────────────────────────────
    lang_labels = [
        f"{LANGUAGES[k]['emoji']}  {LANGUAGES[k]['label']}"
        for k in LANGUAGE_KEYS
    ]

    # Persist the selected index so it survives reruns without jumping to 0.
    stored_idx = st.session_state.get("language_index", 0)

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_label = st.selectbox(
            "🗂️ Select topic / language:",
            options=lang_labels,
            index=stored_idx,
            key="language_select",
        )
    new_index = lang_labels.index(selected_label)
    lang_key  = LANGUAGE_KEYS[new_index]
    lang_info = LANGUAGES[lang_key]

    with col2:
        st.metric(label="Active topic", value=lang_info["label"])

    # Persist the new index and rerun immediately when the language changes so
    # the rest of the page (data load, history) sees the correct language.
    if new_index != stored_idx:
        st.session_state["language_index"] = new_index
        st.rerun()

    st.divider()

    # ── Load knowledge base ──────────────────────────────────────────────────
    lang_pairs, fallback_pairs = get_language_pairs(lang_key)

    # ── Reset chat history when language changes ─────────────────────────────
    if st.session_state.get("active_lang_key") != lang_key:
        st.session_state["active_lang_key"] = lang_key
        st.session_state["messages"] = []

    # ── Seed welcome message ─────────────────────────────────────────────────
    if not st.session_state.get("messages"):
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    f"Hi! I'm **Study Buddy** — focused on "
                    f"**{lang_info['label']}** {lang_info['emoji']}  \n"
                    "Ask me about syntax, variables, loops, functions, and more. "
                    "I'll also check my General CS knowledge if needed, "
                    "and fall back to an AI model for anything else!"
                ),
                "is_ai": False,
            }
        ]

    # ── Render conversation history ──────────────────────────────────────────
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("is_ai"):
                st.caption("🤖 Answered by Hugging Face AI model")

    # ── Chat input (Streamlit pins this bar to the viewport bottom) ──────────
    user_input = st.chat_input(f"Ask a {lang_info['label']} question…")

    if user_input:
        _handle_user_input(user_input, lang_pairs, fallback_pairs, lang_info)
        st.rerun()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("ℹ️ About")
        st.write(
            "**Study Buddy** uses a **hybrid approach**:  \n"
            "1. **Language KB** — keyword + fuzzy matching in the selected "
            "language's Q&A dataset.  \n"
            "2. **General CS KB** — fallback to the General CS dataset.  \n"
            "3. **AI fallback** — Hugging Face free Inference API for anything "
            "outside the local datasets."
        )

        st.divider()
        st.subheader("📊 Loaded datasets")
        if lang_key != "general":
            st.success(
                f"{lang_info['emoji']} {lang_info['label']}: "
                f"{len(lang_pairs)} pairs"
            )
            st.info(f"💻 General CS fallback: {len(fallback_pairs)} pairs")
        else:
            st.success(f"💻 General CS: {len(lang_pairs)} pairs")

        st.divider()
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            st.success("🔑 HF_TOKEN detected — higher AI rate limit active")
        else:
            st.info(
                "ℹ️ No HF_TOKEN set. AI fallback works without a token "
                "but may hit rate limits. Set `HF_TOKEN` in your environment."
            )

        st.divider()
        st.subheader("🗂️ Languages available")
        for k in LANGUAGE_KEYS:
            info = LANGUAGES[k]
            marker = " ← active" if k == lang_key else ""
            st.markdown(f"- {info['emoji']} **{info['label']}**{marker}")

        st.divider()
        if st.button("🗑️ Clear chat"):
            st.session_state["messages"] = []
            st.rerun()


if __name__ == "__main__":
    main()
