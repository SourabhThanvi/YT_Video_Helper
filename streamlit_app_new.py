import streamlit as st
from rag_backend_new import get_transcript, create_vector_store, get_answer

# Custom CSS for Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
        padding-top: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #16a085;
        padding-top: 20px;
    }
    .stButton button {
        background-color: #16a085;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stSelectbox select {
        padding: 10px;
        font-size: 16px;
    }
    .stTextInput input {
        padding: 10px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-title">YouTube Video Q&A (RAG)</p>', unsafe_allow_html=True)

# Video Type Selection with Description
video_type = st.selectbox(
    "Select Video Type",
    ["general", "educational", "interview", "technical", "summary"],
    index=0,
    help="Choose the type of video to optimize responses"
)

# Show description of selected video type
video_type_descriptions = {
    "general": "General purpose YouTube videos",
    "educational": "Tutorials, lectures, courses, and instructional content",
    "interview": "Podcasts, interviews, discussions with multiple speakers",
    "technical": "Programming tutorials, coding sessions, technical explanations",
    "summary": "Generate a concise summary of the video's main points and key takeaways"
}
st.caption(f"**{video_type.capitalize()}**: {video_type_descriptions[video_type]}")

# Language Selection
language_display = st.selectbox(
    "Select Transcript Language",
    ["English", "Hindi", "French"],
    index=0
)

# Map display name to language code
language_map = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr"
}
language_code = language_map[language_display]

# YouTube URL input
url = st.text_input("YouTube Video URL", placeholder="Enter YouTube URL here")

# Load Video Button
if st.button("Load Video"):
    if url:
        with st.spinner("Fetching transcript and building index..."):
            try:
                transcript_text = get_transcript(url, language_code)
                st.success("Transcript fetched successfully!")
                faiss_index = create_vector_store(transcript_text)
                st.session_state["faiss_index"] = faiss_index
                st.session_state["video_type"] = video_type
                st.success("Video loaded and indexed!")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")

# Ask Questions Section (Visible after loading video)
if "faiss_index" in st.session_state:
    st.markdown('<p class="section-header">Ask Questions</p>', unsafe_allow_html=True)
    st.write(f"Video type: **{st.session_state['video_type']}**")

    question = st.text_input("Ask a question about the video", placeholder="Type your question here...")

    if question:
        with st.spinner("Generating answer..."):
            try:
                # Pass both the question and video type to get_answer
                answer = get_answer(
                    question, 
                    st.session_state["faiss_index"], 
                    st.session_state["video_type"]
                )
                st.markdown(f"### Answer:\n{answer}")
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
