from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

def basic_chat(question: str) -> str:
    """
    Basic chat agent — handles general questions
    that don't require document context.
    """
    prompt = f"""You are a helpful AI assistant. 
Answer the following question clearly and concisely.

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    return response.content