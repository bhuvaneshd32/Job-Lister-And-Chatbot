from pinecone import Pinecone

# Initialize Pinecone
api_key = "pcsk_61MNxg_KG1zYAfQh9M3LSEaXjiwBvnTck97mNPRMsFNW5DCbWY1AvDYiR3AirJNytjTHkS"  
pc = Pinecone(api_key=api_key)

index_name = "jobrecommendation"  # Your Pinecone index name

# Connect to the index
index = pc.Index(index_name)

# Get index statistics
index_stats = index.describe_index_stats()

# Print total vector count
print(f"Total job vectors stored: {index_stats['total_vector_count']}")