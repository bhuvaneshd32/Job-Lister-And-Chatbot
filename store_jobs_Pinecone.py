import time
from pinecone import Pinecone, ServerlessSpec
from preprocess_jobs import fetch_jobs, preprocess_jobs, encode_jobs

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
raw_jobs = fetch_jobs("Machine Learning Engineer OR  Data Analyst", "Remote")  
cleaned_jobs = preprocess_jobs(raw_jobs)  
job_vectors = encode_jobs(cleaned_jobs)  

# Prepare job metadata and embeddings
vectors_to_store = []
for i, job in enumerate(cleaned_jobs):  
    metadata = {
        "title": job["title"],
        "location": job["location"],
        "snippet": job["description"],  
        "salary": job["salary"],
        "source": "jooble.org",
        "link": job["link"],
        "company": job["company"],
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S.0000000"),  
        "id": str(i)
    }
    vectors_to_store.append((str(i), job_vectors[i].tolist(), metadata))

# Upsert data into Pinecone
index.upsert(vectors=vectors_to_store)

print(f"✅ {len(job_vectors)} job embeddings stored in Pinecone!")