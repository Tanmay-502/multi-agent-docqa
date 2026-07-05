from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from agents.basic_agent import basic_chat
from agents.doc_reader_agent import answer_from_document
from agents.question_gen_agent import generate_questions
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    user_question: str
    agent_type: str
    context: Optional[str]
    answer: Optional[str]
    generate_questions: Optional[bool]

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

def router_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    generate_q = state.get("generate_questions", False)
    
    if generate_q:
        return {"agent_type": "question_gen"}
    
    prompt = f"""You are a router. Based on the question below, decide which agent should handle it.

If the question is about a document, PDF, or file content → reply: document
If it's a general question or casual chat → reply: chat

Question: {question}

Reply with only one word (document or chat):"""
    
    llm = get_llm()
    response = llm.invoke(prompt)
    agent_type = response.content.strip().lower()
    
    if agent_type not in ["document", "chat"]:
        agent_type = "chat"
    
    return {"agent_type": agent_type}

def basic_chat_node(state: AgentState) -> AgentState:
    answer = basic_chat(state["user_question"])
    return {"answer": answer}

def document_agent_node(state: AgentState) -> AgentState:
    answer = answer_from_document(
        state["user_question"],
        state.get("context", "No document provided.")
    )
    return {"answer": answer}

def question_gen_node(state: AgentState) -> AgentState:
    context = state.get("context", "No document provided.")
    questions = generate_questions(context)
    return {"answer": questions}

def route_decision(state: AgentState) -> str:
    return state["agent_type"]

def build_graph():
    graph = StateGraph(AgentState)
    
    graph.add_node("router", router_node)
    graph.add_node("basic_chat", basic_chat_node)
    graph.add_node("document_agent", document_agent_node)
    graph.add_node("question_gen", question_gen_node)
    
    graph.set_entry_point("router")
    
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "chat": "basic_chat",
            "document": "document_agent",
            "question_gen": "question_gen"
        }
    )
    
    graph.add_edge("basic_chat", END)
    graph.add_edge("document_agent", END)
    graph.add_edge("question_gen", END)
    
    return graph.compile()

agent_graph = build_graph()