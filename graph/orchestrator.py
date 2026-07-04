from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================
# 1. DEFINE THE STATE
# ============================================
class AgentState(TypedDict):
    user_question: str
    agent_type: str
    context: Optional[str]
    answer: Optional[str]

# ============================================
# 2. INITIALIZE THE LLM
# ============================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ============================================
# 3. DEFINE THE NODES (agents)
# ============================================

# Node 1 — Router: decides which agent runs
def router_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    
    prompt = f"""You are a router. Based on the question below, decide which agent should handle it.
    
If the question is about a document, PDF, or file → reply with exactly: document
If it's a general question or casual chat → reply with exactly: chat

Question: {question}

Reply with only one word (document or chat):"""
    
    response = llm.invoke(prompt)
    agent_type = response.content.strip().lower()
    
    if agent_type not in ["document", "chat"]:
        agent_type = "chat"
    
    return {"agent_type": agent_type}

# Node 2 — Basic Chat Agent
def basic_chat_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    
    response = llm.invoke(question)
    
    return {"answer": response.content}

# Node 3 — Document Agent
def document_agent_node(state: AgentState) -> AgentState:
    question = state["user_question"]
    context = state.get("context", "No document provided.")
    
    prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    
    return {"answer": response.content}

# ============================================
# 4. ROUTING FUNCTION
# ============================================
def route_decision(state: AgentState) -> str:
    return state["agent_type"]

# ============================================
# 5. BUILD THE GRAPH
# ============================================
def build_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("basic_chat", basic_chat_node)
    graph.add_node("document_agent", document_agent_node)
    
    # Set entry point
    graph.set_entry_point("router")
    
    # Add conditional edges from router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "chat": "basic_chat",
            "document": "document_agent"
        }
    )
    
    # Both agents lead to END
    graph.add_edge("basic_chat", END)
    graph.add_edge("document_agent", END)
    
    return graph.compile()

# Create the graph instance
agent_graph = build_graph()