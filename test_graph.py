from graph.orchestrator import agent_graph

# Test 1 - General question (should go to basic_chat)
print("=== Test 1: General Question ===")
result = agent_graph.invoke({
    "user_question": "What is the capital of India?",
    "agent_type": "",
    "context": None,
    "answer": None
})
print(f"Agent used: {result['agent_type']}")
print(f"Answer: {result['answer']}")

print("\n=== Test 2: Document Question ===")
result = agent_graph.invoke({
    "user_question": "What does the document say about machine learning?",
    "agent_type": "",
    "context": "Machine learning is a subset of AI that enables systems to learn from data.",
    "answer": None
})
print(f"Agent used: {result['agent_type']}")
print(f"Answer: {result['answer']}")