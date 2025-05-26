import streamlit as st
import requests
import os

# Configuration
API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(page_title="Equity Research Tool", layout="wide")

st.title("📰 Generative AI Equity Research Tool")

# Sidebar inputs
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker (e.g. AAPL)", value="AAPL")
num = st.sidebar.slider("Number of Filings to Ingest", min_value=1, max_value=5, value=2)

# Ingest & classify
if st.sidebar.button("Ingest & Classify"):
    with st.spinner("Fetching & classifying filings..."):
        resp = requests.post(f"{API_URL}/ingest", json={"ticker": ticker, "count": num})
        if resp.status_code == 200:
            ingested = resp.json()["ingested"]
            st.success("Ingestion complete!")
            for item in ingested:
                st.subheader(f"{item['form']} on {item['date']}")
                st.write("**Component:**", item['component'])
        else:
            st.error(f"Error ingesting: {resp.text}")

# Summarization UI
st.header("Summarize Text")
input_text = st.text_area("Paste filing text or summary input here:", height=200)
if st.button("Generate Summary"):
    if input_text.strip():
        with st.spinner("Summarizing..."):
            resp = requests.post(f"{API_URL}/summarize", json={"text": input_text})
            if resp.status_code == 200:
                summary = resp.json().get("summary")
                st.subheader("Executive Summary & Key Highlights")
                st.write(summary)
            else:
                st.error(f"Error summarizing: {resp.text}")
    else:
        st.error("Please provide text to summarize.")

# Q&A UI
st.header("Ask a Question")
question = st.text_input("Enter your question about the filings:")
if st.button("Get Answer"):
    if question.strip():
        with st.spinner("Retrieving answer..."):
            resp = requests.post(f"{API_URL}/ask", json={"text": question})
            if resp.status_code == 200:
                answer = resp.json().get("answer")
                st.subheader("Answer")
                st.write(answer)
            else:
                st.error(f"Error retrieving answer: {resp.text}")
    else:
        st.error("Please enter a question.")
