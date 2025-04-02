import streamlit as st
import redis
from pdf_parser import extract_text_from_pdf, extract_resume_info
from preprocess_jobs import fetch_jobs, preprocess_jobs, encode_jobs
import json
import time
import pandas as pd

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

st.title("AI Resume Parser & Job Recommendation System")

uploaded_files = st.file_uploader(
    "Upload Resumes", type=["pdf"], accept_multiple_files=True, key="unique_key_1")

if uploaded_files:
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    extracted_info = [extract_resume_info(text) for text in resumes]
    
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Extracted Info": extracted_info})
    st.write(results)
    
    skill_query = " OR ".join(extracted_info)
    user_query = st.text_input("Enter job-related query:")
    cache_message = st.empty()  
    cache_key = f"job_recs:{skill_query}"
    
  
    cached_jobs = redis_client.get(cache_key)
    if cached_jobs:
        st.success("Loaded job recommendations from cache!")
        cleaned_jobs = json.loads(cached_jobs)
    else:
        st.info("Fetching job recommendations...")
        jobs = fetch_jobs(skill_query, "Remote", page=1)
        cleaned_jobs = preprocess_jobs(jobs)
        redis_client.setex(cache_key, 3600, json.dumps(cleaned_jobs))  
        st.success("Job recommendations cached!")
    
    st.header("Recommended Jobs")
    for job in cleaned_jobs[:5]:
        st.subheader(job.get("title", "No Title"))
        st.write(job.get("company", "Anonymous"))
        st.write(job.get("description", "No Description"))
        st.write(f"[Apply Here]({job.get('link', '#')})")
    
    if st.button("Clear Cache"):
        redis_client.delete(cache_key)
        st.success("Cache cleared!")
