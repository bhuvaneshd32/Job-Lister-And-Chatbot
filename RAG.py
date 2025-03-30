import time
import html
import uuid
import re
import os
import torch
from pinecone import Pinecone
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from preprocess_jobs import fetch_jobs, preprocess_jobs, encode_jobs

# Load API Key & Region securely
api_key = os.getenv("PINECONE_API_KEY")  # Set via environment variable
region = os.getenv("PINECONE_REGION")  # Example: "us-west4-gcp"

if not api_key or not region:
    raise ValueError("❌ Missing Pinecone API Key or Region! Set PINECONE_API_KEY and PINECONE_REGION in env.")

# Initialize Pinecone with region
pc = Pinecone(api_key=api_key)
index_name = "jobrecommendation"

# Ensure the index exists
if index_name not in pc.list_indexes():
    print("❌ Index not found. Please run the job ingestion script first.")
    exit()

index = pc.Index(index_name, pool_threads=4, region=region)  # Added `region`

# Load models efficiently
embedder = SentenceTransformer("all-MiniLM-L6-v2")

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

# Function to clean text
def clean_text(text):
    text = html.unescape(text)  # Decode HTML entities
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces & newlines
    text = text.replace("nbsp", "").strip()  # Remove non-breaking space artifacts
    return text

# Function to retrieve relevant job listings
def retrieve_context(user_query, top_k=5):
    embedding = embedder.encode(user_query).tolist()
    
    try:
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        retrieved_docs = [
            clean_text(match.get("metadata", {}).get("snippet", "No relevant job description available."))
            for match in results.get("matches", [])
        ]
        return retrieved_docs if retrieved_docs else ["No relevant job listings found."]
    
    except Exception as e:
        print(f"❌ Error retrieving data from Pinecone: {e}")
        return ["No relevant job listings found."]

# Function to generate response
def generate_rag_response(user_query):
    retrieved_docs = retrieve_context(user_query)

    if "No relevant job listings found." in retrieved_docs:
        return "I couldn't find any relevant job listings."

    formatted_context = "\n".join(retrieved_docs)
    prompt = f"Given the following job descriptions, answer the query accurately:\n\n{formatted_context}\n\nQuery: {user_query}\n\nResponse:"

    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output_tokens = model.generate(**input_tokens, max_new_tokens=150)
    
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return response if response.strip() else "I'm unable to generate a relevant response."

# Interactive loop for querying
if __name__ == "__main__":
    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break
        response = generate_rag_response(user_query)
        print("\n🤖 Generated Response:", response)