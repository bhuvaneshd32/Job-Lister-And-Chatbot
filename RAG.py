import pinecone
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

# Initialize Pinecone
PINECONE_API_KEY = "pcsk_61MNxg_KG1zYAfQh9M3LSEaXjiwBvnTck97mNPRMsFNW5DCbWY1AvDYiR3AirJNytjTHkS"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "jobrecommendation"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Load SentenceTransformer for Query Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load T5 Model for Response Generation
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Retrieve relevant jobs from Pinecone
def retrieve_context(user_query, top_k=5):
    query_embedding = embedder.encode(user_query).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    print("\n🔍 Raw Pinecone Results:", results)  # Debugging print

    retrieved_docs = [
        match["metadata"].get("text", "No text found") 
        for match in results.get("matches", []) if "metadata" in match
    ]
    
    print("📄 Retrieved Documents:", retrieved_docs)  # Debugging print
    return retrieved_docs

# Generate response using retrieved jobs
def generate_rag_response(user_query):
    retrieved_docs = retrieve_context(user_query, top_k=8)

    if not retrieved_docs:
        return "I'm sorry, but I couldn't find relevant job listings for your query."

    formatted_context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{formatted_context}\n\nQuestion: {user_query}\n\nAnswer:"

    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output_tokens = model.generate(**input_tokens, max_new_tokens=150)

    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return response

# User Input Loop for Queries
def chat():
    while True:
        user_query = input("\nEnter your job search query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Goodbye! 👋")
            break
        response = generate_rag_response(user_query)
        print("\n🤖 Generated Response:", response)

if __name__ == "__main__":
    chat()