import time
import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Initialize Pinecone
api_key = "pcsk_61MNxg_KG1zYAfQh9M3LSEaXjiwBvnTck97mNPRMsFNW5DCbWY1AvDYiR3AirJNytjTHkS"
pc = Pinecone(api_key=api_key)

index_name = "jobrecommendation"  # Replace with your index name
index = pc.Index(index_name)

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model for generating embeddings

# Initialize the text generation model (Hugging Face's GPT-2 for RAG)
generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

# Caching dictionary for storing query results
query_cache = {}

# Function to retrieve jobs based on query
def retrieve_jobs(query, top_k=5):
    """Retrieve the top-k most relevant job postings based on the query."""
    query_vector = model.encode([query])[0]  # Convert query to vector
    query_vector = query_vector.tolist()  # Convert numpy array to list
    
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

# Function to generate response using RAG (Retrieval-Augmented Generation)
def generate_rag_response(query):
    """Generate a response using the retrieved job listings."""
    # Get the most relevant job postings from Pinecone
    results = get_cached_or_query_results(query, top_k=5)
    
    # Collect the job details (titles, descriptions, etc.) for context
    job_details = "\n".join([f"Job: {match['metadata']['title']} at {match['metadata']['company']} in {match['metadata']['location']}\nDescription: {match['metadata']['snippet']}" for match in results])
    
    # Prepare context for the generation model
    context = f"Job search query: {query}\nRelevant job listings:\n{job_details}\n\nGenerate a summary or response based on these job listings."
    
    # Generate a response based on the context
    response = generator(context, max_new_tokens=50, num_return_sequences=1, truncation=True)
    
    return response[0]['generated_text']

# Example usage of the query system
def test_query():
    user_query = input("Enter job description query: ")
    generated_response = generate_rag_response(user_query)
    
    print("\nGenerated response:")
    print(generated_response)

# This can be called from another script or directly in your main workflow
if __name__ == "__main__":
    test_query()