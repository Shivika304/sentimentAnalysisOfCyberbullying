# app/app_streamlit.py

import sys
from pathlib import Path

# so Python can find src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sentimentAnalysisOfCyberbullying.src.predict_binary import predict_text_binary
import streamlit as st

st.set_page_config(page_title="Cyberbullying Detector", page_icon="⚠")

st.title("🚨 Cyberbullying Detection on Social Media")
st.write("Enter a tweet or comment to check if it may contain cyberbullying.")

text = st.text_area("Enter text here:", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        label = predict_text_binary(text)
        if label == "bullying":
            st.error("⚠ Detected as **CYBERBULLYING**")
        else:
            st.success("✅ Detected as **NOT CYBERBULLYING**")