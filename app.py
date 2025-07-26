# app.py
import streamlit as st
import os
from datetime import datetime

from rag_utils import (
    create_vectorstore_from_url,
    build_qa_chain,
    build_summary_chain,
    generate_chat_pdf_buffer,
    create_vectorstore_from_text
)
from pdf_utils import (
    process_pdf,
    build_pdf_qa_chain,
    build_pdf_summary_chain
)
from process_youtube import process_youtube_video


st.set_page_config(page_title="Multi-Source RAG QA", layout="wide")
st.title("MiRAG")
st.subheader("Multi-input Retrieval-Augmented Generation System")
# Initialize session state
for key in [
    "vectorstore", "chain", "history", "raw_docs", "max_memory",
    "pdf_chain", "pdf_history", "pdf_docs"
]:
    if key not in st.session_state:
        st.session_state[key] = [] if "history" in key or "docs" in key else None
st.session_state.max_memory = 3


# Tabs for Web, PDF, YouTube, and Custom Text QA
tab_custom, tab_web, tab_pdf, tab_youtube = st.tabs([
    "Custom Text QA", "Web QA", "PDF QA", "YouTube QA"
])
# tab_custom, tab_web, tab_pdf, tab_youtube = st.tabs([
#     "   Custom Text QA   ",
#     "       Web QA       ",
#     "       PDF QA       ",
#     "    YouTube QA      "
# ])


# --- 📝 Custom Text QA Tab ---
with tab_custom:
    st.header("Custom Text-Based QA")

    # Initialize default chatbot (no vectorstore) if not already done
    if "default_manual_chain" not in st.session_state:
        try:
            st.session_state.default_manual_chain = build_qa_chain(None)  # Pass None or use an LLM-only chain
            st.session_state.manual_history = []
        except Exception as e:
            st.error(f"Failed to initialize default chatbot: {e}")

    st.markdown("You can chat directly, or optionally paste custom text to build a RAG-based QA system.")

    with st.form("manual_text_form"):
        manual_text = st.text_area("Paste your text below (optional)", height=150,
                                   placeholder="Leave blank to use general chatbot. Paste any large text for RAG.")
        submitted = st.form_submit_button("Process Text")

        if submitted:
            if manual_text.strip():
                with st.spinner("Processing input text and building vectorstore..."):
                    try:
                        vs = create_vectorstore_from_text(manual_text)
                        st.session_state.manual_chain = build_qa_chain(vs)
                        st.session_state.manual_history = []
                        st.success("Custom text processed and RAG chain created.")
                    except Exception as e:
                        st.error(f"Failed to process text: {e}")
            else:
                st.session_state.manual_chain = st.session_state.default_manual_chain
                st.session_state.manual_history = []
                st.info("Using default chatbot mode (no custom text).")

    # Use either the default or custom chain
    current_chain = st.session_state.get("manual_chain", st.session_state.get("default_manual_chain"))

    if current_chain:
        st.subheader("Ask a Question")
        manual_q = st.text_input("Your Question", key="manual_q")

        if st.button("Get Answer", key="manual_a") and manual_q:
            with st.spinner("Thinking..."):
                try:
                    memory_context = "\n\n".join(
                        [f"Q: {q}\nA: {a}" for q, a in st.session_state.manual_history[-st.session_state.max_memory:]]
                    )
                    full_q = f"{memory_context}\n\n{manual_q}" if memory_context else manual_q
                    answer = current_chain.invoke(full_q)
                    st.session_state.manual_history.append((manual_q, answer))
                except Exception as e:
                    st.error(f"Error: {e}")

        for q, a in st.session_state.manual_history[::-1]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

        # --- Optional Download Button ---
        if st.session_state.get("manual_history"):
            st.subheader("Download Chat History")
            try:
                buffer = generate_chat_pdf_buffer(st.session_state["manual_history"])
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"manual_chat_history_{timestamp}.pdf"

                st.download_button(
                    label="Download Chat History PDF",
                    data=buffer,
                    file_name=filename,
                    mime="application/pdf",
                    key="download_manual_pdf_btn"
                )
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")


# --- 🌐 Web URL QA Tab ---
with tab_web:
    st.header("Web-Based RAG Question Answering")

    with st.form("url_form"):
        url = st.text_input("Enter URL to analyze", placeholder="https://...")
        use_selenium = st.checkbox("Use Selenium Loader")
        submitted = st.form_submit_button("Load URL and Build Vectorstore")

        if submitted and url:
            with st.spinner("Loading content and building vectorstore..."):
                try:
                    vs, docs = create_vectorstore_from_url(url, use_selenium)
                    st.session_state.vectorstore = vs
                    st.session_state.chain = build_qa_chain(vs)
                    st.session_state.raw_docs = docs
                    st.session_state.history = []
                    st.success("Vectorstore successfully created.")
                except Exception as e:
                    st.error(f"Failed to process URL: {e}")

    if st.session_state.vectorstore and st.session_state.chain:
        st.subheader("Ask a Question")
        question = st.text_input("Your Question", key="web_q")

        if st.button("Get Answer", key="web_a") and question:
            with st.spinner("Thinking..."):
                try:
                    memory_context = "\n\n".join(
                        [f"Q: {q}\nA: {a}" for q, a in st.session_state.history[-st.session_state.max_memory:]]
                    )
                    full_query = f"{memory_context}\n\n{question}" if memory_context else question
                    answer = st.session_state.chain.invoke(full_query)
                    st.session_state.history.append((question, answer))
                except Exception as e:
                    st.error(f"Error: {e}")

        for q, a in st.session_state.history[::-1]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

        st.subheader("Summarize Webpage")
        if st.button("Summarize Page"):
            with st.spinner("Summarizing..."):
                try:
                    docs_text = "\n\n".join(doc.page_content for doc in st.session_state.raw_docs)
                    summary_chain = build_summary_chain()
                    summary = summary_chain.invoke({"context": docs_text})
                    st.markdown("### Summary")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
    else:
        st.warning("⚠️ Load a URL first.")

# --- 📄 PDF QA Tab ---
with tab_pdf:
    st.header("PDF-Based QA & Summarization")

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        with st.spinner("Processing PDF..."):
            try:
                pdf_text, pdf_docs = process_pdf(pdf_file)
                st.session_state.pdf_chain = build_pdf_qa_chain(pdf_docs)
                st.session_state.pdf_docs = pdf_docs
                st.session_state.pdf_history = []
                st.success("PDF processed successfully.")
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")

    if st.session_state.get("pdf_chain"):
        st.subheader("Ask a Question About PDF")
        pdf_q = st.text_input("Your PDF Question", key="pdf_q")

        if st.button("Get PDF Answer", key="pdf_a") and pdf_q:
            with st.spinner("Thinking..."):
                try:
                    memory_context = "\n\n".join(
                        [f"Q: {q}\nA: {a}" for q, a in st.session_state.pdf_history[-st.session_state.max_memory:]]
                    )
                    full_q = f"{memory_context}\n\n{pdf_q}" if memory_context else pdf_q
                    answer = st.session_state.pdf_chain.invoke(full_q)
                    st.session_state.pdf_history.append((pdf_q, answer))
                except Exception as e:
                    st.error(f"Error: {e}")

        # Display chat history
        for q, a in st.session_state.pdf_history[::-1]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

        # --- Download PDF Chat History Button ---
        if st.session_state.get("pdf_history"):
            st.subheader("Download PDF Chat History")

            try:
                buffer = generate_chat_pdf_buffer(st.session_state["pdf_history"])
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"chat_history_{timestamp}.pdf"

                # Make this button stateless by giving a unique key and putting it inside an if-block
                st.download_button(
                    label="Download Chat History PDF",
                    data=buffer,
                    file_name=filename,
                    mime="application/pdf",
                    key="download_pdf_btn"
                )
            except Exception as e:
                st.error(f"❌ Failed to generate PDF: {e}")

        # --- Summarize PDF ---
        st.subheader("Summarize PDF")
        if st.button("Summarize PDF"):
            with st.spinner("Summarizing PDF..."):
                try:
                    full_pdf = "\n\n".join(doc.page_content for doc in st.session_state.pdf_docs)
                    summary = build_pdf_summary_chain().invoke({"context": full_pdf})
                    st.markdown("### Summary")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
    else:
        st.info("Upload a PDF to begin.")

# --- 📺 YouTube QA Tab ---
with tab_youtube:
    st.header("YouTube Video QA & Summarization")

    yt_url = st.text_input("Enter YouTube Video URL")

    if st.button("Load Transcript"):
        with st.spinner("Fetching and processing transcript..."):
            try:
                vs, yt_docs = process_youtube_video(yt_url)
                st.session_state.yt_chain = build_qa_chain(vs)
                st.session_state.yt_docs = yt_docs
                st.session_state.yt_history = []
                st.success("Transcript loaded and vectorstore created.")
            except Exception as e:
                st.error(f"❌ Failed to process video: {e}")

    if st.session_state.get("yt_chain"):
        st.subheader("Ask a Question About Video")
        yt_q = st.text_input("Your Question", key="yt_q")

        if st.button("Get Answer", key="yt_a") and yt_q:
            with st.spinner("Thinking..."):
                try:
                    memory_context = "\n\n".join([f"Q: {q}\nA: {a}"
                                                  for q, a in st.session_state.yt_history[-st.session_state.max_memory:]])
                    full_q = f"{memory_context}\n\n{yt_q}" if memory_context else yt_q
                    answer = st.session_state.yt_chain.invoke(full_q)
                    st.session_state.yt_history.append((yt_q, answer))
                except Exception as e:
                    st.error(f"Error: {e}")

        for q, a in st.session_state.yt_history[::-1]:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
        # --- Download YouTube Chat History Button ---
        if st.session_state.get("yt_history"):
            st.subheader("Download YouTube Chat History")

            try:
                buffer = generate_chat_pdf_buffer(st.session_state["yt_history"])
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"youtube_chat_history_{timestamp}.pdf"

                st.download_button(
                    label="📥 Download Chat History PDF",
                    data=buffer,
                    file_name=filename,
                    mime="application/pdf",
                    key="download_yt_pdf_btn"
                )
            except Exception as e:
                st.error(f"❌ Failed to generate PDF: {e}")

        st.subheader("Summarize Video Transcript")
        if st.button("Summarize Video"):
            with st.spinner("Summarizing..."):
                try:
                    full_text = "\n\n".join(doc.page_content for doc in st.session_state.yt_docs)
                    summary = build_summary_chain().invoke({"context": full_text})
                    st.markdown("### Summary")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
