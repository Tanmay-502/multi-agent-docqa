# orchestrator.py
# The brain of the multi-agent system
# Builds and manages the LangGraph workflow that routes
# user queries to the correct agent

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from agents.basic_agent import basic_chat
from agents.doc_reader_agent import answer_from_document
from agents.question_gen_agent import generate_questions
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================
# 1. AGENT STATE
# Shared dictionary that flows through the graph
# Every node reads from and writes to this state
# ============================================
class AgentState(TypedDict):
    user_question: str        # The question from the user
    agent_type: str           # Which agent to use (set by router)
    context: Optional[str]    # Document chunks retrieved from FAISS
    answer: Optional[str]     # Final answer (set by whichever agent runs)
    generate_questions: Optional[bool]  # Flag to trigger question gen agent

# ============================================
# 2. LLM INSTANCE
# Shared across all nodes in the graph
# ============================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ============================================
# 3. NODE DEFINITIONS
# Each node is a function that takes state
# and returns an updated state
# ============================================

def router_node(state: AgentState) -> AgentState:
    """
    Router node — reads the question and decides
    which agent should handle it.
    Uses LLM to classify: 'document', 'chat', or 'question_gen'
    """
    question = state["user_question"]
    generate_q = state.get("generate_questions", False)
    
    # If generate_questions flag is set, skip routing
    if generate_q:
        return {"agent_type": "question_gen"}
    
    # Ask LLM to classify the question
    prompt = f"""You are a router. Based on the question below, decide which agent should handle it.

If the question is about a document, PDF, or file content → reply: document
If it's a general question or casual chat → reply: chat

Question: {question}

Reply with only one word (document or chat):"""
    
    response = get_llm().invoke(prompt)
    agent_type = response.content.strip().lower()
    
    # Default to chat if unexpected response
    if agent_type not in ["document", "chat"]:
        agent_type = "chat"
    
    return {"agent_type": agent_type}

def basic_chat_node(state: AgentState) -> AgentState:
    """Calls basic_chat agent and stores answer in state"""
    answer = basic_chat(state["user_question"])
    return {"answer": answer}

def document_agent_node(state: AgentState) -> AgentState:
    """Calls document agent with question + retrieved context"""
    answer = answer_from_document(
        state["user_question"],
        state.get("context", "No document provided.")
    )
    return {"answer": answer}

def question_gen_node(state: AgentState) -> AgentState:
    """Calls question generation agent with document context"""
    context = state.get("context", "No document provided.")
    questions = generate_questions(context)
    return {"answer": questions}

# ============================================
# 4. ROUTING FUNCTION
# Reads agent_type from state and returns
# the name of the next node to run
# ============================================
def route_decision(state: AgentState) -> str:
    return state["agent_type"]

# ============================================
# 5. BUILD THE GRAPH
# Wires nodes together with edges
# ============================================
def build_graph():
    graph = StateGraph(AgentState)
    
    # Register all nodes
    graph.add_node("router", router_node)
    graph.add_node("basic_chat", basic_chat_node)
    graph.add_node("document_agent", document_agent_node)
    graph.add_node("question_gen", question_gen_node)
    
    # Entry point — always starts at router
    graph.set_entry_point("router")
    
    # Conditional edges — router decides which node runs next
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "chat": "basic_chat",
            "document": "document_agent",
            "question_gen": "question_gen"
        }
    )
    
    # All agents lead to END
    graph.add_edge("basic_chat", END)
    graph.add_edge("document_agent", END)
    graph.add_edge("question_gen", END)
    
    # Compile and return the runnable graph
    return graph.compile()

# Create the graph instance — imported by app.py and api/main.py
agent_graph = build_graph()