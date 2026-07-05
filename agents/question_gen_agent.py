from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

def generate_questions(context: str, num_questions: int = 5) -> str:
    prompt = f"""You are an expert educator and examiner.
Based on the context below, generate {num_questions} clear exam questions.

Format:
Q1. [question]
Q2. [question]
Q3. [question]
Q4. [question]
Q5. [question]

Context:
{context}

Generate {num_questions} exam questions:"""
    
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content