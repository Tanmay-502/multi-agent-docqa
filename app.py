import streamlit as st
from dotenv import load_dotenv
import os
from agents.doc_reader_agent import process_document
from graph.orchestrator import agent_graph

load_dotenv()

# Load from Streamlit secrets if available
try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except:
    pass

st.set_page_config(page_title="DocQA Chatbot", page_icon="📄")
st.title("📄 Multi-Agent Document Q&A")
st.write("Upload a PDF and ask questions about it.")

# Sidebar for API key input
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input(
        "Enter Groq API Key:",
        type="password",
        help="Get your free key at console.groq.com"
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("API Key set!")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses:
    - 🦙 **Groq LLaMA 3.1** — LLM
    - 🔗 **LangGraph** — Agent orchestration
    - 📊 **FAISS** — Vector search
    - ⚡ **FastAPI** — Backend API
    """)

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("⚠️ Please enter your Groq API key in the sidebar first!")
    else:
        with st.spinner("Processing document..."):
            vector_store, num_chunks, num_chars = process_document(uploaded_file)

        st.write(f"✅ Total characters: {num_chars}")
        st.write(f"✅ Split into {num_chunks} chunks")
        st.success("✅ Document processed and ready!")

        # Chat interface
        st.subheader("💬 Ask questions about your document")
        user_question = st.text_input("Type your question here:")

        if user_question:
            with st.spinner("Finding answer..."):
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                relevant_docs = retriever.invoke(user_question)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

                result = agent_graph.invoke({
                    "user_question": user_question,
                    "agent_type": "",
                    "context": context,
                    "answer": None,
                    "generate_questions": False
                })
                st.write("### Answer:")
                st.write(result["answer"])
                st.caption(f"Agent used: {result['agent_type']}")

        # Generate Questions
        st.divider()
        st.subheader("🎯 Generate Exam Questions")
        st.write("Click below to auto-generate exam questions from your document.")

        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
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
                st.write("### Generated Questions:")
                st.write(result["answer"])
                st.caption("Agent used: question_gen")