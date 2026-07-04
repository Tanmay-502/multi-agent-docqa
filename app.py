import streamlit as st
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="DocQA Chatbot", page_icon="📄")
st.title("📄 Multi-Agent Document Q&A")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully!")

    # Extract text from PDF
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text()

    st.write(f"✅ Extracted {len(pdf_reader.pages)} pages")
    st.write(f"✅ Total characters: {len(raw_text)}")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(raw_text)
    st.write(f"✅ Split into {len(chunks)} chunks")

    # Create embeddings and store in FAISS
    with st.spinner("Processing document..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        vector_store = FAISS.from_texts(chunks, embeddings)
    st.success("✅ Document processed and ready!")

    # Chat interface
    st.subheader("💬 Ask questions about your document")
    user_question = st.text_input("Type your question here:")

    if user_question:
        with st.spinner("Finding answer..."):
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.invoke(user_question)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                api_key=os.getenv("GROQ_API_KEY")
            )

            prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {user_question}

Answer:"""

            response = llm.invoke(prompt)
            st.write("### Answer:")
            st.write(response.content)