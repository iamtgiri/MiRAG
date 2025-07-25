# process_youtube.py

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

import re

def extract_video_id(url):
    """Extract video ID from various YT URL formats."""
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
    return match.group(1) if match else None

def fetch_yt_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry["text"] for entry in transcript])

def process_youtube_video(url: str):
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    full_text = fetch_yt_transcript(video_id)
    docs = [Document(page_content=full_text)]

    # Split for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(full_text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts, embeddings)

    return vectorstore, docs
