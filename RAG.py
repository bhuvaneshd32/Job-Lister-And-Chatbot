import time
import uuid
import pinecone
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
PINECONE_API_KEY ="pcsk_61MNxg_KG1zYAfQh9M3LSEaXjiwBvnTck97mNPRMsFNW5DCbWY1AvDYiR3AirJNytjTHkS"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "jobrecommendation"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Load Embedding Model for Querying Pinecone
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load T5 model for text generation
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def retrieve_context(user_query, top_k=5):
    """Retrieve relevant job descriptions from Pinecone."""
    embedding = embedder.encode(user_query).tolist()
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)

    # Extract job descriptions (snippet field) safely
    retrieved_docs = [
        html.unescape(match["metadata"].get("snippet", "No text found"))  # Decodes HTML entities
        for match in results["matches"]
        if "metadata" in match
    ]
    return retrieved_docs

def generate_rag_response(user_query):
    """Generate an AI response using retrieved job descriptions."""
    retrieved_docs = retrieve_context(user_query)
    
    if not retrieved_docs or all(doc == "No text found" for doc in retrieved_docs):
        return "I'm sorry, but I couldn't find relevant job listings for your query."
    
    formatted_context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{formatted_context}\n\nQuestion: {user_query}\n\nAnswer:"
    
    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_tokens = model.generate(**input_tokens, max_new_tokens=150)
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    return response

# Interactive Query Loop
if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        response = generate_rag_response(user_query)
        print("\n🤖 Generated Response:", response)
