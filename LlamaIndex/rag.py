import json
import os
import numpy as np
import os
from urllib.parse import urlparse

import requests

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
print(f"Nearest words: {nearest_words[0]} and {nearest_words[1]}")
print(f"Distance: {distances[nearest_words]:.4f}")
print(f"Farthest words: {farthest_words[0]} and {farthest_words[1]}")
print(f"Distance: {distances[farthest_words]:.4f}")

# Set the embed model and llm
Settings.embed_model = embed_model
Settings.llm = llm

print("Downloading source files...")

source_files = [
    "https://docs.aws.amazon.com/lambda/latest/dg/lambda-dg.pdf",
    "https://docs.aws.amazon.com/vpc-lattice/latest/ug/vpc-lattice.pdf",
    "https://docs.aws.amazon.com/vpc-lattice/latest/APIReference/vpc-lattice-api.pdf",
]

# Create the data directory if it doesn't exist
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)
    # Download source files
    for url in source_files:
        # Get the filename from the URL 
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(DATA_DIR, filename)
        
        # Download the file
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download: {url}")

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    print("Loading data and creating index...")
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    print("Loading existing index...")
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

print("Querying index...")

query = "Can I use VPC Lattice with a Lambda function?"
print(f"Query: {query}")

query_engine = index.as_query_engine()
response = query_engine.query(query)
print(f"Response: {response}")
