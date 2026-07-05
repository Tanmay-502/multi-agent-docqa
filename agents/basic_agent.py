from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

def basic_chat(question: str) -> str:
    prompt = f"""You are a helpful AI assistant. 
Answer the following question clearly and concisely.

Question: {question}

Answer:"""
    
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content