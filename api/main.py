from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.orchestrator import agent_graph
from agents.doc_reader_agent import process_document
import io

app = FastAPI(
    title="Multi-Agent DocQA API",
    description="API for document Q&A and question generation",
    version="1.0.0"
)

# Allow Streamlit to talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Store vector store in memory
vector_store_cache = {}

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================
class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"
    generate_questions: bool = False

class AnswerResponse(BaseModel):
    answer: str
    agent_used: str
    session_id: str

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "message": "Multi-Agent DocQA API is running!",
        "endpoints": ["/upload", "/ask", "/generate-questions", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF and process it into a vector store."""
    try:
        # Read file content
        content = await file.read()
        pdf_file = io.BytesIO(content)
        pdf_file.name = file.filename

        # Process document
        vector_store, num_chunks, num_chars = process_document(pdf_file)

        # Cache the vector store
        vector_store_cache["default"] = vector_store

        return {
            "message": "Document processed successfully!",
            "filename": file.filename,
            "num_chunks": num_chunks,
            "num_chars": num_chars
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the uploaded document."""
    try:
        context = ""

        # Get context from vector store if available
        if request.session_id in vector_store_cache:
            vector_store = vector_store_cache[request.session_id]
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.invoke(request.question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Run through agent graph
        result = agent_graph.invoke({
            "user_question": request.question,
            "agent_type": "",
            "context": context,
            "answer": None,
            "generate_questions": False
        })

        return AnswerResponse(
            answer=result["answer"],
            agent_used=result["agent_type"],
            session_id=request.session_id
        )
    except Exception as e:
        return AnswerResponse(
            answer=f"Error: {str(e)}",
            agent_used="error",
            session_id=request.session_id
        )

@app.post("/generate-questions")
async def generate_questions(session_id: str = "default"):
    """Generate exam questions from the uploaded document."""
    try:
        if session_id not in vector_store_cache:
            return {"error": "No document uploaded. Please upload a PDF first."}

        vector_store = vector_store_cache[session_id]
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.invoke("main topics and key concepts")
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        result = agent_graph.invoke({
            "user_question": "Generate exam questions",
            "agent_type": "",
            "context": context,
            "answer": None,
            "generate_questions": True
        })

        return {
            "questions": result["answer"],
            "agent_used": result["agent_type"]
        }
    except Exception as e:
        return {"error": str(e)}