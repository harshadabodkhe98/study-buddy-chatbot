# How IBM Bob Was Used in This Project

## Project: Study Buddy — AI Chatbot

IBM Bob, an AI coding agent integrated into a VS Code-based IDE, was used extensively throughout the development of this project.

## Key Workflows

### 1. Initial Project Setup
Bob generated the complete initial project structure from a single prompt, including:
- `app.py` — terminal-based chatbot
- `app_web.py` — Streamlit web interface
- `data.txt` — Q&A knowledge base
- `requirements.txt` and `README.md`

### 2. Accuracy Improvements
Bob improved the chatbot's matching engine by adding:
- Synonym and abbreviation support (e.g., "AI" matching "artificial intelligence")
- Plural/singular handling (e.g., "loops" matching "loop")
- Fuzzy matching for typos and casual phrasing

### 3. Language-Specific Support
Bob added multi-language support, creating separate Q&A datasets for Python, Java, C, and C++, along with a language selector in both the terminal and web interfaces.

### 4. AI-Powered Fallback
Bob integrated a Hugging Face AI model as a fallback, so the chatbot can answer questions outside its local dataset, with graceful error handling if the API is unavailable.

### 5. Debugging and Refactoring
Bob was used to identify and fix runtime errors (e.g., Streamlit widget state issues), refactor the codebase into a shared `chatbot_engine.py` module used by both interfaces, and validate that all features worked correctly after each change.

### 6. Deployment Support
Bob helped prepare the project for deployment, ensuring all dependencies were correctly listed and the code was clean and well-commented for hosting on Streamlit Community Cloud.

## Summary
Bob's agentic capabilities — planning multi-step tasks, writing and editing code across multiple files, and testing changes — significantly accelerated development, enabling this project to go from an initial idea to a fully working, deployed AI chatbot within the hackathon timeframe.