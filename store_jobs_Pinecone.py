import time
from pinecone import Pinecone, ServerlessSpec
from preprocess_jobs import fetch_jobs, preprocess_jobs, encode_jobs
import uuid

# Initialize Pinecone
api_key = "pcsk_61MNxg_KG1zYAfQh9M3LSEaXjiwBvnTck97mNPRMsFNW5DCbWY1AvDYiR3AirJNytjTHkS"
pc = Pinecone(api_key=api_key)

index_name = "jobrecommendation"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=384, 
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

# Fetch and preprocess jobs dynamically
raw_jobs = fetch_jobs("ML Intern", "Remote")  
cleaned_jobs = preprocess_jobs(raw_jobs)  
job_vectors = encode_jobs(cleaned_jobs)

# Prepare job metadata and embeddings
vectors_to_store = []
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
    job_id = str(uuid.uuid4())  # Generates a unique ID
    vectors_to_store.append((job_id, vector.tolist(), metadata))

# Upsert data into Pinecone
index.upsert(vectors=vectors_to_store)


print(f"✅ {len(job_vectors)} job embeddings stored in Pinecone!")