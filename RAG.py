import pinecone
#import torch
import numpy as np
from rank_bm25 import BM25Okapi
import re
from bs4 import BeautifulSoup
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
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
"""
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
"""
GEMINI_API_KEY = "AIzaSyCgt7gn1MaoJrOjSzaybX4BpWx4rSZiR1U"
genai.configure(api_key=GEMINI_API_KEY)

#print("Hello")
# Function to clean job descriptions
def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")  # Remove HTML tags
    text = soup.get_text()
    text = text.replace("\xa0", " ").replace("\n", " ").replace("nbsp", "").strip()
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces & newlines
    #print(text)
    return text

def retrieve_context(query, top_k=5):

    pinecone_results = index.query(
        vector=embedder.encode(query).tolist(),
        top_k=50,  # Retrieve more than top_k for better ranking
        include_metadata=True
    )

    #print(pinecone_results)
    job_dict = {
        match["id"]: {
            "company": match["metadata"].get("company","N/A"),
            "title": match["metadata"].get("title", "N/A"),
            "salary":match["metadata"].get("salary", "N/A"),
            "location": match["metadata"].get("location", "N/A"),
            "link":match["metadata"].get("link","N/A"),
            "description": clean_text(match["metadata"].get("snippet", ""))  # Cleaned
        }
        for match in pinecone_results["matches"]
    }
    
    # Tokenize job titles + descriptions for BM25
    tokenized_jobs = [(job["title"] + " " + job["description"]).split() for job in job_dict.values()]
    bm25 = BM25Okapi(tokenized_jobs)

    # BM25 Retrieval
    bm25_scores = bm25.get_scores(query.split())
    job_ids = list(job_dict.keys())  # Maintain ID order
    bm25_results = [(job_ids[i], bm25_scores[i]) for i in np.argsort(bm25_scores)[::-1][:top_k]]

    # Apply Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    rrf_k = 60  

    # BM25 RRF scoring
    for rank, (idx, score) in enumerate(bm25_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank)

    # Pinecone RRF scoring
    for rank, match in enumerate(pinecone_results["matches"], start=1):
        idx = match["id"]
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank)

    # Sort results by RRF score
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [job_dict[idx] for idx, _ in sorted_results[:top_k] if idx in job_dict]  

# Generate response using retrieved jobs
def generate_rag_response(user_query):
    retrieved_docs = retrieve_context(user_query, top_k=8)

    if not retrieved_docs:
        return "I'm sorry, but I couldn't find relevant job listings for your query."

    formatted_jobs = "\n\n".join(
        f"**{doc['title']}**\n"
        f"**Company:** {doc['company']}\n"
        f"**Location:** {doc['location']}\n"
        f"**Salary:** {doc['salary']}\n"
        f"**Link:** {doc['link']}\n"
        f"**Description:** {doc['description']}"
        for doc in retrieved_docs
    )

    #print(formatted_jobs)

    prompt = f"""
You are an AI-powered job assistant. Your task is to extract **only** the relevant job listings from the data provided.

### **User Query:**
"{user_query}"

### **Extracted Job Listings:**
{formatted_jobs}

### **Instructions:**
1. **DO NOT** generate jobs that are not listed in the extracted job listings above.
2. **DO NOT** make up companies like Google, Amazon, etc.
3. **ONLY** reformat the given job listings and present them in the correct format.
4. Present the job listings in a clear, structured paragraph format.
5. Use bullet points to separate different jobs.
6. Write in full sentences and avoid unnecessary technical details.
7. DO NOT output JSON or structured data—only human-readable text.

---
""".strip()

    # Tokenization for LLaMA 2
    """
    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    output_tokens = model.generate(**input_tokens, max_new_tokens=1024)
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return response
    """
    model = genai.GenerativeModel("gemini-1.5-pro")  # Use Gemini 1.5 Pro
    response = model.generate_content(prompt)

    return response.text if response else "Failed to generate response from Gemini."

# User Input Loop for Queries
def chat():
    while True:
        user_query = input("\nEnter your job search query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Goodbye! ")
            break
        response = generate_rag_response(user_query)
        print("\nGenerated Response:", response)

if __name__ == "__main__":
    chat()