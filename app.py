# app.py

import streamlit as st
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
from youtube_utils import process_youtube_video


# --- App Configuration ---
st.set_page_config(
    page_title="MiRAG",
    layout="wide",
    page_icon="🧠"
)

# --- App Header ---
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>MiRAG</h1>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: center; color: gray; margin-top: 0;'>"
    "Multi-input Retrieval-Augmented Generation System"
    "</h4>",
    unsafe_allow_html=True
)

# Initialize session state
for key in [
    "vectorstore", "chain", "history", "raw_docs", "max_memory",
    "pdf_chain", "pdf_history", "pdf_docs"
]:
    if key not in st.session_state:
        st.session_state[key] = [] if "history" in key or "docs" in key else None
st.session_state.max_memory = 3



# ---------------------------------------------------------------------
page = st.selectbox(
    label="Select a data source for Retrieval-Augmented Question Answering:",
    options=["Custom Text QA", "Web QA", "PDF QA", "YouTube QA"],
    index=0,
    format_func=lambda x: x,  # Keep proper case for readability
)

# ---------------------------------------------------------------------
# --- 📝 Custom Text QA Tab ---
if page == "Custom Text QA":
    st.header("Custom Text-Based QA")

    # Initialize default chain if needed (no vectorstore)
    if "default_manual_chain" not in st.session_state:
        try:
            st.session_state.default_manual_chain = build_qa_chain(None)
            st.session_state.manual_history = []
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {e}")

    st.markdown("Chat directly or optionally paste custom text to build a context-aware QA system.")
    
    use_custom_text = st.checkbox("Use Custom Text for RAG", value=False)

    if use_custom_text:
        with st.form("manual_text_form"):
            manual_text = st.text_area(
                "Paste your text below",
                height=150,
                placeholder="Leave blank to use general chatbot. Paste any large text for RAG."
            )
            submitted = st.form_submit_button("Process Text")

            if submitted:
                with st.spinner("Processing input text..."):
                    try:
                        if manual_text.strip():
                            vs = create_vectorstore_from_text(manual_text)
                            st.session_state.manual_chain = build_qa_chain(vs)
                            st.success("Custom text processed and QA chain created.")
                        else:
                            st.session_state.manual_chain = st.session_state.default_manual_chain
                            st.info("No text provided. Using default chatbot.")
                        st.session_state.manual_history = []
                    except Exception as e:
                        st.error(f"Failed to process text: {e}")

    # Use the appropriate QA chain
    current_chain = st.session_state.get("manual_chain") or st.session_state.get("default_manual_chain")

    if current_chain:
        st.subheader("Ask a Question")
        manual_q = st.text_input("Your Question", key="manual_q")

        if st.button("Get Answer", key="manual_a") and manual_q:
            with st.spinner("Generating answer..."):
                try:
                    memory_context = "\n\n".join(
                        [f"Q: {q}\nA: {a}" for q, a in st.session_state.manual_history[-st.session_state.max_memory:]]
                    )
                    full_q = f"{memory_context}\n\n{manual_q}" if memory_context else manual_q
                    answer = current_chain.invoke(full_q)
                    st.session_state.manual_history.append((manual_q, answer))
                except Exception as e:
                    st.error(f"Error: {e}")

        # Display past Q&A
        for q, a in reversed(st.session_state.manual_history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

        # Optionally offer PDF download of chat
        if st.session_state.manual_history:
            st.subheader("Download Chat History")
            try:
                buffer = generate_chat_pdf_buffer(st.session_state.manual_history)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"manual_chat_history_{timestamp}.pdf"

                st.download_button(
                    label="Download Chat History as PDF",
                    data=buffer,
                    file_name=filename,
                    mime="application/pdf",
                    key="download_manual_pdf_btn"
                )
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")


# --- 🌐 Web URL QA Tab ---
elif page == "Web QA":
    st.header("Web-Based RAG Question Answering")

    with st.form("url_form"):
        url = st.text_input("Enter URL to analyze", placeholder="https://...")
        use_selenium = st.checkbox("Use Selenium Loader (for javascript-heavy pages)", value=False)
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
elif page == "PDF QA":
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
elif page == "YouTube QA":
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

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; color: gray;'>
    <a href='https://github.com/iamtgiri/MiRAG' target='_blank' style='text-decoration: none; color: inherit;'><strong>MiRAG</strong></a>
         is an open-source project developed by 
        <a href='https://github.com/iamtgiri' target='_blank' style='text-decoration: none; color: inherit;'> Tanmoy Giri</a>.
    </div>
    """,
    unsafe_allow_html=True
)
