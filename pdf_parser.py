from PyPDF2 import PdfReader
import pandas as pd
import requests
import spacy
import streamlit as st 



# Load NLP model
nlp = spacy.load("en_core_web_sm")

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



st.title("AI Resume Parser & Job Recommendation System")

uploaded_files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    resume_texts = [extract_resume_info(text) for text in resumes]
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Extracted Info": resume_texts})
    #st.write(results)