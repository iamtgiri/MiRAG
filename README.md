
# 🔍 MiRaGS — Multi-input Retrieval-Augmented Generation System

**MiRaGS** is an interactive, multi-modal application built with Streamlit that leverages Retrieval-Augmented Generation (RAG) to perform question-answering and summarization across various content types:

- 🌐 Web pages  
- 📄 PDF documents  
- 📺 YouTube videos  

Built on LangChain, Gemini (Google Generative AI), and FAISS, MiRaGS enables users to query unstructured content intelligently and intuitively.

---

## 🚀 Features

### 🔹 Web QA (RAG from URLs)
- Extracts and embeds content from any public URL (JS and non-JS).
- Allows conversational QA and summarization.
- Supports streaming memory for context-aware answers.

### 🔹 PDF QA
- Upload any PDF and perform:
  - Context-aware Q&A
  - Full document summarization
  - Chat history download as PDF

### 🔹 YouTube Video QA
- Enter a YouTube URL to fetch its transcript.
- Ask questions and generate a concise summary.
- Ideal for educational content and long-form media.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** – UI
- **LangChain** – Chain and embedding orchestration
- **Google Generative AI (Gemini)** – Language model & embeddings
- **FAISS** – Vectorstore for semantic retrieval
- **youtube-transcript-api** – Transcript extraction
- **fpdf** – PDF generation for downloadable chat logs

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iamtgiri/MiRaGS.git
   cd MiRaGS
````

2. **Create virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables:**

   ```bash
   export GOOGLE_API_KEY=your_api_key_here
   ```

5. **Run the app:**

   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

```
MiRaGS/
├── app.py                      # Main Streamlit app
├── pdf_utils.py
├── process_youtube.py  
├── rag_utils.py               # All utility functions & chain builders
└── README.md
├── requirements.txt
```

---

## 📸 Screenshots

> (Include screenshots or gifs if you want visual documentation)

---

## ✅ To-Do / Roadmap

* [ ] OpenAI / Claude / Mistral model switch
* [ ] Combined PDF + Web + YT memory context
* [ ] Session-level chat export
* [ ] Multilingual support
* [ ] UI enhancements with avatars and themes

---

## 🧠 Credits

* Built with [LangChain](https://www.langchain.com/)
* Powered by [Google Gemini](https://ai.google.dev/)
* PDF export via [FPDF](https://pyfpdf.github.io/)
* Transcription via [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)

---

## 📄 License

MIT License © 2025 MiRaGS Contributors

---


