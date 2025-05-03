# streamlit_app.py
import streamlit as st
from rag_backend import get_transcript, create_vector_store, get_answer

st.title("YouTube Video Q&A (RAG)")
url = st.text_input("YouTube Video URL")
if st.button("Load Video"):
    if url:
        with st.spinner("Fetching transcript and building index..."):
            transcript_text = get_transcript(url)
            st.success("Transcript fetched!")
            faiss_index = create_vector_store(transcript_text)
            st.session_state["faiss_index"] = faiss_index
        st.success("Video loaded and indexed!")

if "faiss_index" in st.session_state:
    question = st.text_input("Ask a question about the video")
    if question:
        with st.spinner("Generating answer..."):
            answer = get_answer(question, st.session_state["faiss_index"])
        st.write("**Answer:**", answer)
