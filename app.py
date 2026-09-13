"""
Study Buddy — Terminal Chatbot
===============================
Loads Q&A pairs from data files and runs an interactive chat loop.

At startup the user picks a programming language (or General CS).
All questions are then answered using that language's dataset, with
the General CS dataset as a fallback when no language-specific match is found.

Matching uses a four-tier hybrid approach (all in chatbot_engine.py):
  1. LOCAL (language) — keyword + stem matching in the chosen language's data
  2. LOCAL (general)  — same matching in data.txt when tier-1 finds nothing
  3. AI               — Hugging Face free Inference API as a last resort
  4. STATIC           — friendly message when all other tiers fail

Run:  python app.py
"""

from chatbot_engine import (
    LANGUAGES,
    LANGUAGE_KEYS,
    load_language_pairs,
    find_best_answer_for_language,
)


# ---------------------------------------------------------------------------
# Language selection menu
# ---------------------------------------------------------------------------

def pick_language() -> str:
    """
    Display the language menu and return the chosen language key.
    Keeps asking until a valid choice is entered.
    """
    print("=" * 60)
    print("  Study Buddy -- CS Knowledge Chatbot")
    print("=" * 60)
    print("\n  Select a topic / language to study:\n")

    for idx, key in enumerate(LANGUAGE_KEYS, start=1):
        info = LANGUAGES[key]
        print(f"  {idx}. {info['emoji']}  {info['label']}")

    print()

    while True:
        try:
            choice = input("  Enter 1-5: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "general"

        if choice.isdigit():
            n = int(choice)
            if 1 <= n <= len(LANGUAGE_KEYS):
                return LANGUAGE_KEYS[n - 1]

        print("  Please enter a number between 1 and 5.")


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat(lang_key: str) -> None:
    """
    Load the chosen language's data and run the interactive chat loop.
    """
    info       = LANGUAGES[lang_key]
    lang_label = f"{info['emoji']}  {info['label']}"

    lang_pairs, fallback_pairs = load_language_pairs(lang_key)

    total = len(lang_pairs) + len(fallback_pairs)
    print(f"\n  Loaded {len(lang_pairs)} {info['label']} pairs"
          + (f" + {len(fallback_pairs)} General CS pairs" if fallback_pairs else "")
          + f"  ({total} total)")
    print(f"\n  Active topic: {lang_label}")
    print("  Powered by keyword matching + Hugging Face AI fallback.")
    print("  Type  'exit'  to quit, or 'switch' to change language.\n")

    while True:
        # ── Get user input ─────────────────────────────────────────────────
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! Keep studying!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Bot: Goodbye! Keep studying!")
            break

        if user_input.lower() == "switch":
            # Allow switching language mid-session
            new_key = pick_language()
            chat(new_key)
            return

        # ── Match and respond ──────────────────────────────────────────────
        response, is_exact, is_ai = find_best_answer_for_language(
            user_input, lang_pairs, fallback_pairs
        )

        # Strip Markdown bold markers — not rendered in terminals
        response_plain = response.replace("**", "")

        # Label the source
        if is_exact:
            source = f"[{info['label']} KB]"
        elif is_ai:
            source = "[AI-powered]"
        else:
            source = "[Fallback]"

        print(f"Bot {source}: {response_plain}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    selected_key = pick_language()
    chat(selected_key)
