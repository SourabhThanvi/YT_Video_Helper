# rag_backend.py
import os
from google import genai  # Google GenAI SDK for Gemini
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders.youtube import YoutubeLoader   # optional, if using LangChain
from langchain_community.vectorstores import FAISS # or another embedding model
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings  # optional, if using HuggingFace models
from langchain.text_splitter import RecursiveCharacterTextSplitter  # optional, if using LangChain
from langchain.prompts import PromptTemplate  # optional, if using LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Gemini client (uses GOOGLE_API_KEY env var by default)
from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key: 
#     raise ValueError("GOOGLE_API_KEY environment variable not set")
client = genai.Client(api_key = 'AIzaSyBKWW-ITmNRUcBZ3R6j7ciTeIyoq838spg')

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    # Simple parsing (or use pytube: YouTube(url).video_id)
    from urllib.parse import parse_qs, urlparse
    query = parse_qs(urlparse(url).query)
    return query.get("v", [None])[0]

def get_transcript(video_url: str) -> str:
    """Download transcript text for the given YouTube URL."""
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    # Fetch transcript segments
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    # Combine segments into a single text
    text = " ".join([entry["text"] for entry in transcript_list])
    return text

def create_vector_store(doc_text: str) -> FAISS:
    """Create a FAISS vector store from the given text."""
    # # Split text into documents (e.g., by paragraphs or fixed chunk size)
    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # chunks = splitter.create_documents(doc_text)
    # # Embed and create FAISS index
    # print(len(chunks))
    # print('\n\n')
    # print(chunks[100])
    # print('----------------')
    # embeddings = HuggingFaceEmbeddings()  
    # faiss_index = FAISS.from_documents(chunks, embeddings)
    # return faiss_index
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_text(doc_text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    # Embed and create FAISS index
    embeddings = HuggingFaceEmbeddings()  
    faiss_index = FAISS.from_documents(docs, embeddings)
    return faiss_index

def get_answer(question: str, faiss_index: FAISS) -> str:
    """Perform a RAG query: retrieve relevant docs and ask Gemini."""
    # Retrieve relevant chunks
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    # Form prompt for Gemini (can be adjusted for system/instructions)
    prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
    retrieved_docs= retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.invoke({"context": context_text, "question": question})

    print(final_prompt)
    # Call Gemini API
    response = client.models.generate_content(contents=final_prompt, model="gemini-2.0-flash") 
    answer = response.text  # or appropriate field
    return answer
