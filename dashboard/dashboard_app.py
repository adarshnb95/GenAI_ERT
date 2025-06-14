import streamlit as st
import requests
import json
import pandas as pd
import os
from pathlib import Path

# Config
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

N2_FIELDS = {"Fund Structure", "Fund Subtype", "NAV Reporting Frequency", "Suitability"}

st.set_page_config(page_title="GenAI Equity Research", layout="wide")
tab_ask, tab_classifier = st.tabs(["Ask a Question", "Document Classifier"])

# Ask a Question Tab
with tab_ask:
    st.title("Generative AI Equity Research Tool")
    question = st.text_area("Enter your question about any company or companies:")
    if st.button("Get Answer", key="ask_button"):
        if not question:
            st.error("Please enter a question.")
        else:
            try:
                resp = requests.post(f"{API_URL}/ask", json={"text": question}, timeout=30)
                if resp.status_code == 200:
                    st.subheader("Answer")
                    answer_obj = resp.json().get("answer")
                    # Check for structured N-2 response
                    if isinstance(answer_obj, dict) and set(answer_obj.keys()) == N2_FIELDS:
                        df = pd.DataFrame(list(answer_obj.items()), columns=["Field", "Value"])
                        st.table(df)
                    else:
                        # Display all other responses as plain text or JSON
                        if isinstance(answer_obj, dict):
                            st.json(answer_obj)
                        else:
                            st.write(answer_obj)
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

# Document Classifier Tab
with tab_classifier:
    st.title("Document Classifier & Summarizer")
    st.markdown("Provide a URL or upload a local document to classify and summarize.")
    url = st.text_input("Enter document URL (PDF/HTML/TXT):", key="doc_url")
    uploaded_file = st.file_uploader(
        "Or upload a document (PDF, TXT, HTML):",
        type=["pdf", "txt", "htm", "html"],
        key="doc_upload"
    )
    if st.button("Classify & Summarize", key="classify_button"):
        # Prepare temp directory
        temp_dir = Path.cwd() / "temp_docs"
        temp_dir.mkdir(exist_ok=True)
        file_path = None

        # Download or save upload
        if url:
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                ext = url.split('.')[-1].lower()
                file_path = temp_dir / f"downloaded_doc.{ext}"
                file_path.write_bytes(resp.content)
            except Exception as e:
                st.error(f"Failed to download file: {e}")
                st.stop()
        elif uploaded_file is not None:
            try:
                ext = uploaded_file.name.split('.')[-1].lower()
                file_path = temp_dir / f"uploaded_doc.{ext}"
                file_path.write_bytes(uploaded_file.getbuffer())
            except Exception as e:
                st.error(f"Failed to save uploaded file: {e}")
                st.stop()
        else:
            st.error("Please provide a URL or upload a document.")
            st.stop()

        # Extract text
        txt = ""
        if file_path.suffix.lower() in ['.htm', '.html', '.txt']:
            try:
                txt = file_path.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                st.error(f"Failed to read document: {e}")
                st.stop()
        elif file_path.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(str(file_path))
                pages = [page.extract_text() or "" for page in reader.pages]
                txt = "\n".join(pages)
            except Exception as e:
                st.error(f"Failed to parse PDF: {e}")
                st.stop()
        else:
            st.error("Unsupported file type.")
            st.stop()

        # Classification
        try:
            from classifier.predict import classify_text
            doc_type = classify_text(txt)
            st.subheader("Document Type")
            st.success(doc_type)
        except Exception as e:
            st.error(f"Classification error: {e}")
            st.stop()

        # Summarization
        try:
            from summarization.summarize import summarize_text
            summary = summarize_text(txt)
            st.subheader("Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Summarization error: {e}")
            st.stop()

        # Industry Trend Analysis
        st.subheader("Industry & Company Trends")
        try:
            from summarization.summarize import answer_question_for_ticker
            trends = answer_question_for_ticker("", "Analyze current market trends for the industry discussed in this document.")
            st.write(trends)
        except Exception as e:
            st.error(f"Trend analysis error: {e}")
