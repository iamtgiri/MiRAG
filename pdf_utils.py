# pdf_utils.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS

import tempfile
from langchain_community.document_loaders import PyPDFLoader

def process_pdf(uploaded_file):
    # Write uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load PDF using path
    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load()

    # Split and return
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(pages)

    return "\n\n".join([doc.page_content for doc in split_docs]), split_docs


# PDF QA
def build_pdf_qa_chain(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template="""You are an expert assistant with deep domain knowledge. Based only on the provided context, generate a detailed, well-structured explanation that fully answers the question. If the context is insufficient to answer, just say you don't know. Use clear reasoning, relevant facts and examples from the context wherever applicable.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3,
        max_output_tokens=600
    )

    parser = StrOutputParser()

    chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda x: "\n\n".join(doc.page_content for doc in x)),
        "question": RunnablePassthrough()
    }) | prompt | model | parser

    return chain

# PDF Summary
def build_pdf_summary_chain():
    summary_prompt = PromptTemplate(
        template="""You are an expert summarization system. Your task is to generate a large, comprehensive and detailed summary of the provided content. The summary must be thorough yet well-structured, retaining all crucial information, context, nuances, and supporting details.
context:
{context}

Summary:""",
        input_variables=["context"]
    )

    summary_model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.3,
        max_output_tokens=2500
    )

    parser = StrOutputParser()
    return summary_prompt | summary_model | parser
