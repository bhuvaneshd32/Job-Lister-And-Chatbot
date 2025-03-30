import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer  # For text embedding

# Initialize Pinecone
api_key = "pcsk_61MNxg_KG1zYAfQh9M3LSEaXjiwBvnTck97mNPRMsFNW5DCbWY1AvDYiR3AirJNytjTHkS"  # Replace with your actual API key
pc = Pinecone(api_key=api_key)

index_name = "jobrecommendation"  # Replace with your index name

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model for generating embeddings

# Example job description to query
query_job_description = "Machine Learning Engineer with experience in deep learning and natural language processing."

# Encode the job description to a vector (this converts the text to a list of floats)
query_vector = model.encode([query_job_description])[0]  # This gives you a vector of floats

# Convert the numpy array to a list
query_vector = query_vector.tolist()

# Check the type of query_vector to make sure it's a list of floats
print(f"Query vector type: {type(query_vector)}")
print(f"Query vector length: {len(query_vector)}")

# Query the index
index = pc.Index(index_name)
response = index.query(
    vector=query_vector, 
    top_k=5,  # Number of results to return
    include_metadata=True  # Include metadata in the results
)

# Print the results
print("Query results:")
for match in response['matches']:
    print(f"Job ID: {match['id']}")
    print(f"Title: {match['metadata']['title']}")
    print(f"Location: {match['metadata']['location']}")
    print(f"Description: {match['metadata']['snippet']}")
    print(f"Link: {match['metadata']['link']}")
    print("-" * 30)