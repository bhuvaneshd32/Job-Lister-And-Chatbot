import time
import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
api_key = "pcsk_61MNxg_KG1zYAfQh9M3LSEaXjiwBvnTck97mNPRMsFNW5DCbWY1AvDYiR3AirJNytjTHkS"
pc = Pinecone(api_key=api_key)

index_name = "jobrecommendation"  # Replace with your index name
index = pc.Index(index_name)

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model for generating embeddings

# Caching dictionary for storing query results
query_cache = {}

# Function to retrieve jobs based on query
def retrieve_jobs(query, top_k=5):
    """Retrieve the top-k most relevant job postings based on the query."""
    query_vector = model.encode([query])[0]  # Convert query to vector
    query_vector = query_vector.tolist()  # Convert numpy array to list
    
    print(f"Query vector type: {type(query_vector)}")
    print(f"Query vector length: {len(query_vector)}")
    
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    
    return response['matches']

# Function to fetch and cache query results
def get_cached_or_query_results(query, top_k=5):
    """Fetch from cache if available, else query the Pinecone index."""
    if query in query_cache:
        print("Fetching from cache")
        return query_cache[query]
    else:
        print("Querying Pinecone")
        results = retrieve_jobs(query, top_k)
        query_cache[query] = results  # Cache the results
        return results

# Example usage of the query system
def test_query():
    user_query = input("Enter job description query: ")
    results = get_cached_or_query_results(user_query, top_k=5)
    
    print("Query results:")
    for match in results:
        print(f"Job ID: {match['id']}")
        print(f"Title: {match['metadata']['title']}")
        print(f"Location: {match['metadata']['location']}")
        print(f"Description: {match['metadata']['snippet']}")
        print(f"Link: {match['metadata']['link']}")
        print("-" * 30)

# Run the test query
test_query()