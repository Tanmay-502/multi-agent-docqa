![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 🤖 Multi-Agent Document Q&A System

A production-ready multi-agent AI system that reads documents and generates 
intelligent answers and exam questions using LangGraph orchestration.

🌐 **Live Demo:** https://multi-agent-docapp.streamlit.app

## 📸 What it does

1. Upload any PDF document
2. Ask questions about it — get AI-powered answers
3. Auto-generate exam questions from the document

## 🏗️ Architecture
User
↓
Streamlit UI (frontend)
↓ HTTP requests
FastAPI Backend (api/main.py)
↓
LangGraph Orchestrator (graph/orchestrator.py)
↓           ↓              ↓
Basic       Doc Reader    Question
Agent       Agent         Generator
↓           ↓              ↓
Groq LLaMA 3.1

## 🛠️ Tech Stack

| Tool | Role |
|---|---|
| **FastAPI** | REST API backend |
| **LangGraph** | Multi-agent orchestration |
| **LangChain** | Agent logic and RAG pipeline |
| **FAISS** | Vector database for document search |
| **HuggingFace** | Free embeddings (all-MiniLM-L6-v2) |
| **Groq LLaMA 3.1** | Free LLM API |
| **Streamlit** | Frontend UI + deployment |

## 🤖 Agents

- **Router Agent** — Classifies question and routes to correct agent
- **Basic Chat Agent** — Handles general questions
- **Document Reader Agent** — RAG-based PDF Q&A
- **Question Generator Agent** — Generates exam questions

## 🚀 Run Locally

```bash
git clone https://github.com/Tanmay-502/multi-agent-docqa
cd multi-agent-docqa
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file:

GROQ_API_KEY=your_key_here

Run FastAPI backend:
```bash
uvicorn api.main:app --reload --port 8000
```

Run Streamlit frontend:
```bash
streamlit run app.py
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/upload` | Upload PDF |
| POST | `/ask` | Ask a question |
| POST | `/generate-questions` | Generate exam questions |

API docs available at: `http://localhost:8000/docs`

## 📅 Built in 7 Days

- [x] Day 1 — Environment setup, LLM connection
- [x] Day 2 — RAG PDF chatbot baseline
- [x] Day 3 — LangGraph multi-agent orchestration
- [x] Day 4 — Individual agents + question generation
- [x] Day 5 — FastAPI backend integration
- [x] Day 6 — Streamlit Cloud deployment
- [x] Day 7 — Polish and documentation