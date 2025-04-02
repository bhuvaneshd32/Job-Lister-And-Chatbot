from pdf_parser import extract_text_from_pdf, extract_resume_info
from preprocess_jobs import fetch_jobs, preprocess_jobs, encode_jobs
from image_parser import extract_text_from_image
from RAG import chat, index
import streamlit as st
import pandas as pd
import pinecone
import time


st.title("AI Resume Parser & Job Recommendation System....")


"""
uploaded_files = st.file_uploader(
    "Upload Resumes (PDF or Image)", 
    type=["pdf", "jpg", "jpeg", "png"], 
    accept_multiple_files=True, 
    key="unique_key_1" 
)
"""
uploaded_files = st.file_uploader(
    "Upload Resumes", 
    type=["pdf"], 
    accept_multiple_files=True, 
    key="unique_key_1" 
)

if uploaded_files:
    
    """
    resumes = []
    
    for file in uploaded_files:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
        else:
            text = extract_text_from_image(file)
        
        resumes.append(text)
    """
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    extracted_info = [extract_resume_info(text) for text in resumes]
    
    
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Extracted Info": extracted_info})
    st.write(results)
    
    
    jobs = fetch_jobs(" OR ".join(extracted_info), "Remote", page=1)
    cleaned_jobs = preprocess_jobs(jobs)
    
    #st.write(cleaned_jobs)
    st.header("Recommended Jobs")
    for cleaned_job in cleaned_jobs[:5]:
        st.subheader(cleaned_job.get("title", "No Title"))
        st.write(cleaned_job.get("company", "Anonymous"))
        #st.write(cleaned_job.get("description", "No Description"))
        st.write(f"[Apply Here]({cleaned_job.get('link', '#')})")
        
    job_vectors = encode_jobs(cleaned_jobs)
    
    vectors_to_store = []
    i=1
    for job, vector in zip(cleaned_jobs, job_vectors):  
        metadata = {
            "title": job["title"],
            "location": job["location"],
            "snippet": job["description"],  
            "salary": job["salary"],
            "source": "jooble.org",
            "link": job["link"],
            "company": job["company"],
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S.0000000"),  
        }
        job_id = str(i)  # Generates a unique ID
        i+=1
        vectors_to_store.append((job_id, vector.tolist(), metadata))

    
    index.upsert(vectors=vectors_to_store)
    
    st.success("Job data stored in Pinecone. You can now ask queries!")
    
    
    user_query = st.text_input("Enter job-related query:")
    if user_query:
        response = chat(user_query)
        st.write(response)
    
    if st.button("Clear Job Data from Pinecone"):
        index.delete(delete_all=True)
        st.success("Pinecone index cleared!")
