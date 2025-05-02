import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import requests
import spacy

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Jooble API credentials
JOOBLE_API_URL = "https://jooble.org/api/8ed62732-bed6-48af-bf13-3eda2a4ad13a"

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    return text

def extract_resume_info(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT"]]
    return " ".join(skills)

def fetch_jobs_from_jooble(query):
    payload = {"keywords": query, "location": "", "page": 1}
    response = requests.post(JOOBLE_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("jobs", [])
    return []

st.title("AI Resume Parser & Job Recommendation System")

uploaded_files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    resume_texts = [extract_resume_info(text) for text in resumes]
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Extracted Info": resume_texts})
    st.write(results)

    # Fetch job recommendations
    top_resume = resume_texts[0]
    jobs = fetch_jobs_from_jooble(top_resume)
    st.header("Recommended Jobs")
    for job in jobs[:5]:
        st.subheader(job.get("title", "No Title"))
        st.write(job.get("snippet", "No Description"))
        st.write(f"[Apply Here]({job.get('link', '#')})")