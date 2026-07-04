# Multi-Agent Document Q&A System

A multi-agent AI system that reads documents and generates questions from them.

## Tech Stack
- **FastAPI** — API backend
- **LangGraph** — Multi-agent orchestration
- **LangChain** — Agent logic and RAG pipeline
- **FAISS** — Vector database for document search
- **Streamlit** — Frontend UI
- **Groq (LLaMA 3.1)** — Free LLM API

## Agents
- **Basic Agent** — Simple chatbot for context understanding
- **Documents Reader Agent** — Reads PDFs using RAG
- **Question Generation Agent** — Generates exam questions from documents

## Progress
- [x] Day 1 — Environment setup, LLM connection working
- [x] Day 2 — PDF chatbot baseline
- [ ] Day 3 — LangGraph core
- [ ] Day 4 — Individual agents
- [ ] Day 5 — FastAPI integration
- [ ] Day 6 — Streamlit frontend + deployment
- [ ] Day 7 — Polish and documentation