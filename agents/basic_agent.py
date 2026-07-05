# basic_agent.py
# Handles general questions that don't require document context
# This is the simplest agent — just takes a question and returns an answer

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the LLM — using Groq's free LLaMA 3.1 model
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

def basic_chat(question: str) -> str:
    """
    Basic chat agent — handles general questions
    that don't require document context.
    
    Args:
        question: The user's question
    Returns:
        str: The LLM's response
    """
    # Create a simple prompt for general questions
    prompt = f"""You are a helpful AI assistant. 
Answer the following question clearly and concisely.

Question: {question}

Answer:"""
    
    # Send to LLM and return the response text
    response = get_llm().invoke(prompt)
    return response.content