import streamlit as st
import requests
import json
import pandas as pd
import os

# Config
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="GenAI Equity Research", layout="wide")
st.title("Generative AI Equity Research Tool")

# User input
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
                # Display dict answers as table
                if isinstance(answer_obj, dict):
                    df = pd.DataFrame(list(answer_obj.items()), columns=["Field", "Value"])
                    st.table(df)
                else:
                    # Try parsing string JSON
                    try:
                        data = json.loads(answer_obj)
                        df = pd.DataFrame(list(data.items()), columns=["Field", "Value"])
                        st.table(df)
                    except Exception:
                        st.write(answer_obj)
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
