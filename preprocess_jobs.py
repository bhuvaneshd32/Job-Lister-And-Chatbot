import requests
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer



API_KEY = "ee033cd6-069b-4bb8-99f6-0855bcc45b5a"
API_URL = f"https://jooble.org/api/{API_KEY}"

def fetch_jobs(keywords, location, page=1):
    """Fetch job listings from Jooble API dynamically based on user input."""
    payload = {
        "keywords": keywords,
        "location": location,
        "page": page
    }
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        return response.json().get("jobs", [])  # Return job listings
    else:
        print("Error:", response.status_code, response.text)
        return []

def clean_text(text):
    """Removes HTML tags, special characters, and extra spaces from text."""
    if not text:
        return "N/A"  
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    return text.strip().lower()  

def preprocess_jobs(jobs):
    """Preprocess job data by cleaning and structuring."""
    cleaned_jobs = []
    for job in jobs:
        cleaned_jobs.append({
            "title": clean_text(job.get("title")),
            "company": clean_text(job.get("company")),
            "location": clean_text(job.get("location")),
            "description": clean_text(job.get("snippet")),  
            "salary": clean_text(job.get("salary")),
            "link": job.get("link", "N/A")  
        })
    return cleaned_jobs


model = SentenceTransformer('all-MiniLM-L6-v2')  

def encode_jobs(jobs):
    """Encodes job descriptions into numerical vectors."""
    descriptions = [job["description"] for job in jobs]  
    vectors = model.encode(descriptions, convert_to_numpy=True)  
    return vectors

# Test preprocessing on the fetched jobs
jobs = fetch_jobs("Data Scientist OR Data Analyst OR Data Engineer","Remote",page=1)
cleaned_jobs = preprocess_jobs(jobs)  # Preprocess first 5 jobs
job_vectors = encode_jobs(cleaned_jobs)

