import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="DocQA Chatbot", page_icon="📄")
st.title("📄 Multi-Agent Document Q&A")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing document..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )
        data = response.json()

    if "error" in data:
        st.error(data["error"])
    else:
        st.success("PDF uploaded successfully!")
        st.write(f"✅ Total characters: {data['num_chars']}")
        st.write(f"✅ Split into {data['num_chunks']} chunks")
        st.success("✅ Document processed and ready!")

        st.subheader("💬 Ask questions about your document")
        user_question = st.text_input("Type your question here:")

        if user_question:
            with st.spinner("Finding answer..."):
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": user_question,
                        "session_id": "default",
                        "generate_questions": False
                    }
                )
                result = response.json()
                st.write("### Answer:")
                st.write(result["answer"])
                st.caption(f"Agent used: {result['agent_used']}")

        st.divider()
        st.subheader("🎯 Generate Exam Questions")
        st.write("Click below to auto-generate exam questions from your document.")

        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                response = requests.post(
                    f"{API_URL}/generate-questions",
                    params={"session_id": "default"}
                )
                result = response.json()
                st.write("### Generated Questions:")
                st.write(result["questions"])
                st.caption("Agent used: question_gen")