from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_questions(context: str, num_questions: int = 5) -> str:
    """
    Question generation agent — generates exam-style
    questions from document content.
    """
    prompt = f"""You are an expert educator and examiner.
Based on the context below, generate {num_questions} clear and insightful exam questions.

Format your response as:
Q1. [question]
Q2. [question]
Q3. [question]
Q4. [question]
Q5. [question]

Context:
{context}

Generate {num_questions} exam questions:"""
    
    response = llm.invoke(prompt)
    return response.content