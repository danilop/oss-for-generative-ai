import os
import random

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
#from haystack.components.builders import PromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator
from haystack import Document, Pipeline

from datasets import load_dataset

document_store = InMemoryDocumentStore()

# The dataset is about the Seven Wonders of the Ancient World.
# It contains textual information about these historical landmarks.
# https://huggingface.co/datasets/bilgeyucel/seven-wonders
dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

# For sentence and short paragraph encoding
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

# For sentence and short paragraph encoding
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

retriever = InMemoryEmbeddingRetriever(document_store)

template = [ChatMessage.from_user("""
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
""")]

#prompt_builder = PromptBuilder(template=template)
prompt_builder = ChatPromptBuilder(template=template)

generator = AmazonBedrockChatGenerator(model="anthropic.claude-3-5-sonnet-20240620-v1:0")

basic_rag_pipeline = Pipeline()

# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

examples = [
    "Where is Gardens of Babylon?",
    "Why did people build Great Pyramid of Giza?",
    "What does Rhodes Statue look like?",
    "Why did people visit the Temple of Artemis?",
    "What is the importance of Colossus of Rhodes?",
    "What happened to the Tomb of Mausolus?",
    "How did Colossus of Rhodes collapse?",
]

question = random.choice(examples)

print(question)

response = basic_rag_pipeline.run({"text_embedder": {"text": question}, "prompt_builder": {"question": question}})

print(response["llm"]["replies"][0].content)
