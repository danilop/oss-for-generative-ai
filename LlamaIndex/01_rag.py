import json
import os
import numpy as np
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding

print("Supported embedding models:")

supported_models = BedrockEmbedding.list_supported_models()

print(json.dumps(supported_models, indent=2))

llm = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0.0,
)

embed_model = BedrockEmbedding(model="amazon.titan-embed-g1-text-02")

print("Testing embedding...")

words = ["latte", "cappuccino", "car"]

print(f"Words: {words}")

# Compute embeddings for all words
embeddings = [np.array(embed_model.get_text_embedding(word)) for word in words]

# Compute all possible distances between words
distances = {}
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        distance = np.linalg.norm(embeddings[i] - embeddings[j])
        distances[(words[i], words[j])] = distance

# Find the nearest and farthest word couples
nearest_words = min(distances, key=distances.get)
farthest_words = max(distances, key=distances.get)

# Print the results
print(f"Nearest words: {nearest_words[0]} and {nearest_words[1]} - Distance: {distances[nearest_words]:.4f}")
print(f"Farthest words: {farthest_words[0]} and {farthest_words[1]} - Distance: {distances[farthest_words]:.4f}")


# Set the embed model and llm
Settings.embed_model = embed_model
Settings.llm = llm

# Check if storage already exists
DATA_DIR = "./data"
PERSIST_DIR = "./storage_rag"
if not os.path.exists(PERSIST_DIR):
    # Load the documents and create the index
    print("Loading data and creating index...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    # Store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    print("Loading existing index...")
    # Load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

print("Querying index...")

query = "Can I use VPC Lattice with a Lambda function?"
print(f"Query: {query}")

query_engine = index.as_query_engine()
response = query_engine.query(query)
print(f"Response: {response}")
