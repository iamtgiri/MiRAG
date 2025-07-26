# rag_utils.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import SeleniumURLLoader, WebBaseLoader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fpdf import FPDF
from datetime import datetime
import os
import tempfile
from dotenv import load_dotenv

from io import BytesIO

def load_with_selenium(urls):
    loader = SeleniumURLLoader(urls=urls)
    return loader.load()

def load_with_webbase(urls):
    loader = WebBaseLoader(web_path=urls)
    return loader.load()

def format_doc(retriever_docs):
    return "\n\n".join(doc.page_content for doc in retriever_docs)

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    max_output_tokens=512
)

summary_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.3,
    max_output_tokens=1500   # You can adjust this
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="""
You are an expert assistant with deep domain knowledge. Based only on the provided context, generate a detailed, well-structured explanation that fully answers the question. If the context is insufficient to answer, just say you don't know. Use clear reasoning, relevant facts and examples from the context wherever applicable.

Context:
{context}

Question: {question}

Detailed Answer:""",
    input_variables=["context", "question"]
)

summary_prompt = PromptTemplate(
    template="""
You are a summarization expert. Given the following content from a webpage, generate a comprehensive, concise summary that captures the key points.

Content:
{context}

Summary:""",
    input_variables=["context"]
)

def create_vectorstore_from_url(url: str, use_selenium: bool = False):
    urls = [url]
    docs = load_with_selenium(urls) if use_selenium else load_with_webbase(urls)
    documents = "\n\n".join(doc.page_content for doc in docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts, embeddings)

    return vectorstore, docs  # also return raw docs for summary

chatprompt = PromptTemplate(
    template="""
You are a helpful and knowledgeable assistant. 
Provide concise, accurate answers based on your understanding.

Question: {question}
Answer:""",
    input_variables=["question"]
)


def build_qa_chain(vectorstore = None):
    if vectorstore is None:
        return chatprompt | model | parser
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_doc),
        "question": RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | model | parser
    return final_chain

def build_summary_chain():
    return summary_prompt | summary_model | parser




def generate_chat_pdf_buffer(chat_history):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for q, a in chat_history:
        pdf.set_font("Arial", style='B', size=12)
        pdf.multi_cell(0, 10, f"Q: {q}")
        pdf.set_font("Arial", style='', size=12)
        pdf.multi_cell(0, 10, f"A: {a}")
        pdf.ln(3)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # or your embedding model
from langchain.schema import Document

def create_vectorstore_from_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_text(text)
    wrapped_docs = [Document(page_content=d) for d in docs]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(wrapped_docs, embeddings)
    return vectorstore
