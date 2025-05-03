import os
from google import genai  # Google GenAI SDK for Gemini
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Initialize Gemini client (uses GOOGLE_API_KEY env var by default)
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
# if not api_key: 
#     raise ValueError("GOOGLE_API_KEY environment variable not set")
client = genai.Client(api_key=key)

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    # Simple parsing (or use pytube: YouTube(url).video_id)
    from urllib.parse import parse_qs, urlparse
    query = parse_qs(urlparse(url).query)
    return query.get("v", [None])[0]

# def get_transcript(video_url: str) -> str:
#     """Download transcript text for the given YouTube URL."""
#     video_id = extract_video_id(video_url)
#     if not video_id:
#         raise ValueError("Invalid YouTube URL")
#     # Fetch transcript segments
#     transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
#     # Combine segments into a single text
#     text = " ".join([entry["text"] for entry in transcript_list])
#     return text

def get_transcript(video_url: str, language_code: str = "en") -> str:
    """Download transcript text for the given YouTube URL in the specified language."""
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
    text = " ".join([entry["text"] for entry in transcript_list])
    return text 


def create_vector_store(doc_text: str) -> FAISS:
    """Create a FAISS vector store from the given text."""
    # Enhanced chunking strategy with better size
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(doc_text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    # Embed and create FAISS index
    embeddings = HuggingFaceEmbeddings()  
    faiss_index = FAISS.from_documents(docs, embeddings)
    return faiss_index

def get_prompt_for_video_type(video_type: str) -> PromptTemplate:
    """Return the appropriate prompt template based on video type."""
    
    # General purpose prompt (default)
    general_prompt = PromptTemplate(
        template="""
        # YouTube Transcript Assistant

        ## Context from Video Transcript
        The following are relevant segments from the YouTube video transcript:
        
        {context}
        
        ## Your Task
        You are an expert assistant specialized in analyzing YouTube content. Answer the user's question based ONLY on the information provided in the transcript segments above.
        
        - Provide specific quotes or timestamps when possible to support your answer
        - Maintain the original meaning and nuance from the transcript
        - If the transcript segments don't contain sufficient information to answer the question confidently, explain what's missing and say you don't know
        - Do not make up information or draw from knowledge outside the provided context
        - If the transcript is unclear or ambiguous, acknowledge this in your response
        
        ## User Question
        {question}
        
        ## Your Response
        """,
        input_variables=["context", "question"]
    )
    
    # Educational/Tutorial video prompt
    educational_prompt = PromptTemplate(
        template="""
        # Tutorial Content Analysis Assistant

        ## Tutorial Transcript Context
        {context}
        
        ## Your Task
        You are an expert tutor helping to extract knowledge from educational videos. Answer the user's question based ONLY on the information provided in the transcript segments.
        
        - Explain concepts clearly and step-by-step as presented in the video
        - If the user is asking about a procedure or method, organize your response as instructions
        - Highlight key terminology and definitions that appear in the transcript
        - If multiple approaches or techniques are mentioned, compare them
        - Include any specific examples mentioned in the transcript
        - If the transcript doesn't contain the answer, explain what's missing and say you don't know
        
        ## User Question
        {question}
        
        ## Your Response
        """,
        input_variables=["context", "question"]
    )
    
    # Interview/Podcast video prompt
    interview_prompt = PromptTemplate(
        template="""
        # Interview Content Analysis Assistant

        ## Interview/Podcast Transcript Context
        {context}
        
        ## Your Task
        You are an expert at analyzing interview and conversation content. Answer the user's question based ONLY on the information provided in the transcript segments.
        
        - Distinguish between different speakers' perspectives and opinions
        - Note when a statement is an opinion versus a factual claim
        - Preserve the tone and nuance of the speakers' words
        - Use direct quotes when possible, citing the speaker
        - Consider the conversational context when interpreting statements
        - If the transcript doesn't contain the answer, explain what's missing and say you don't know
        
        ## User Question
        {question}
        
        ## Your Response
        """,
        input_variables=["context", "question"]
    )
    
    # Technical/Coding video prompt
    technical_prompt = PromptTemplate(
        template="""
        # Technical Content Analysis Assistant

        ## Technical Video Transcript Context
        {context}
        
        ## Your Task
        You are an expert at analyzing technical and programming content. Answer the user's question based ONLY on the information provided in the transcript segments.
        
        - If code examples are described, format them properly in code blocks
        - Clarify technical terminology as it's used in the specific context of the video
        - Distinguish between conceptual explanations and specific implementation details
        - Note any warnings, best practices, or common pitfalls mentioned
        - If multiple approaches are discussed, compare their trade-offs as presented
        - If the transcript doesn't contain the answer, explain what's missing and say you don't know
        
        ## User Question
        {question}
        
        ## Your Response
        """,
        input_variables=["context", "question"]
    )
    
    # Summary prompt for condensing video content
    summary_prompt = PromptTemplate(
        template="""
        # Video Summarization Assistant

        ## Video Transcript Segments
        {context}
        
        ## Your Task
        You are an expert at summarizing video content. Create a comprehensive yet concise summary of the video based ONLY on the transcript segments provided.
        
        - Focus on the main topics, key points, and essential takeaways
        - Organize information in a logical structure (introduction, main points, conclusion)
        - Highlight any important facts, statistics, or quotes mentioned 
        - Include the central argument or message of the content
        - Maintain neutrality and accuracy while condensing the information
        - If the transcript segments seem incomplete or disconnected, acknowledge the limitations of your summary
        - Do not add information beyond what's explicitly stated in the transcript
        
        ## User Request
        {question}
        
        ## Your Summary
        """,
        input_variables=["context", "question"]
    )
    
    # Return the appropriate prompt based on video type
    video_type = video_type.lower()
    if "summary" in video_type or "summarize" in video_type or "overview" in video_type:
        return summary_prompt
    elif "tutorial" in video_type or "educational" in video_type or "lecture" in video_type or "course" in video_type:
        return educational_prompt
    elif "interview" in video_type or "podcast" in video_type or "conversation" in video_type or "discussion" in video_type:
        return interview_prompt
    elif "technical" in video_type or "coding" in video_type or "programming" in video_type or "development" in video_type:
        return technical_prompt
    else:
        return general_prompt
    

def get_answer(question: str, faiss_index: FAISS, video_type: str = "general") -> str:
    """Perform a RAG query: retrieve relevant docs and ask Gemini."""
    # Retrieve relevant chunks (increased k for more context)
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # Get the appropriate prompt template based on video type
    prompt = get_prompt_for_video_type(video_type)
    
    # Retrieve documents
    retrieved_docs = retriever.invoke(question)
    
    # Combine retrieved contexts
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Create the final prompt with the context and question
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    
    # Call Gemini API with temperature control for more factual responses
    response = client.models.generate_content(
        contents=final_prompt, 
        model="gemini-2.0-flash"
    ) 
    
    answer = response.text
    return answer