from pdf_parser import extract_text_from_pdf, extract_resume_info
from preprocess_jobs import fetch_jobs, preprocess_jobs, encode_jobs
from RAG import chat, index
import streamlit as st
import pandas as pd
import pinecone
import time
import redis
import json
import hashlib

# Initialize Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

st.title("AI Resume Parser & Job Recommendation System")

# Upload resumes
uploaded_files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True, key="unique_key_1")

if uploaded_files:
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    extracted_info = [extract_resume_info(text) for text in resumes]

    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Extracted Info": extracted_info})
    st.write(results)

    # Generate a hash for caching job recommendations based on resume content
    resume_text_concat = "".join(resumes)  
    resume_hash = hashlib.md5(resume_text_concat.encode()).hexdigest()
    job_cache_key = f"job_recommendations:{resume_hash}"

    # Check cache for job recommendations
    cached_jobs = redis_client.get(job_cache_key)

    if cached_jobs:
        st.success("Loaded job recommendations from cache!")
        cleaned_jobs = json.loads(cached_jobs)
    else:
        st.info("Fetching new job recommendations...")
        skill_query = " OR ".join(extracted_info)
        jobs = fetch_jobs(skill_query, "Remote", page=1)
        cleaned_jobs = preprocess_jobs(jobs)

        # Cache job recommendations for 24 hours
        redis_client.setex(job_cache_key, 86400, json.dumps(cleaned_jobs))
        st.success("Job recommendations cached!")

    # Display job recommendations
    st.header("Recommended Jobs")
    for job in cleaned_jobs[:5]:
        st.subheader(job.get("title", "No Title"))
        st.write(job.get("company", "Anonymous"))
        st.write(f"[Apply Here]({job.get('link', '#')})")

    # Encoding jobs for Pinecone
    job_vectors = encode_jobs(cleaned_jobs)
    vectors_to_store = []
    
    for i, (job, vector) in enumerate(zip(cleaned_jobs, job_vectors), start=1):  
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
        job_id = str(i)
        vectors_to_store.append((job_id, vector.tolist(), metadata))

    index.upsert(vectors=vectors_to_store)
    
    st.success("Job data stored in Pinecone. You can now ask queries!")

    # Query input field
    user_query = st.text_input("Enter job-related query:")

    if user_query:
        start_time = time.time() 
        # Create a unique query cache key using PDF names + query text
        pdf_names = "_".join(file.name for file in uploaded_files)  # Concatenate filenames
        query_cache_key = f"query_response:{pdf_names}:{user_query}"  # Unique cache key

        # Check if response is in cache
        cached_response = redis_client.get(query_cache_key)
        if cached_response:
            st.success("Loaded response from cache!")
            response = cached_response  # Use cached response
        else:
            st.info("Fetching new response...")
            response = chat(user_query)  # Get response from RAG model
            redis_client.setex(query_cache_key, 3600, response)  # Cache for 1 hour
            st.success("Response cached!")

        st.write(response)
        end_time = time.time()  # End timing after query execution
        response_time = round(end_time - start_time, 3)  # Compute time in seconds

        st.write(response)
        st.info(f"⏱️ Query Response Time: {response_time} seconds")

    # Separate buttons for clearing cache
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Query Cache"):
            # Delete only query response keys
            query_keys = redis_client.keys("query_response:*")  
            for key in query_keys:
                redis_client.delete(key)
            st.success("Cleared all query responses from cache!")

    with col2:
        if st.button("Clear Job Cache"):
            # Delete only job recommendation keys
            job_keys = redis_client.keys("job_recommendations:*")
            for key in job_keys:
                redis_client.delete(key)
            st.success("Cleared all job recommendations from cache!")

    if st.button("Clear Job Data from Pinecone"):
        index.delete(delete_all=True)
        st.success("Pinecone index cleared!")
