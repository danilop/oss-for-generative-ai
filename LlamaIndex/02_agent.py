import os

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.tools import FunctionTool

AWS_REGION = "us-east-1"


def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)


embed_model = BedrockEmbedding(model="amazon.titan-embed-g1-text-02")

query_llm = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name=AWS_REGION,
)

PERSIST_DIR = "./storage_agent"
if not os.path.exists(PERSIST_DIR):
    # Load data
    print("Loading data...")
    years = [2021]
    uber_docs = SimpleDirectoryReader(
        input_files=[f"./data/10k/uber_{year}.pdf" for year in years]
    ).load_data()
    # Build index
    print("Building index...")
    uber_index = VectorStoreIndex.from_documents(
        uber_docs,
        embed_model=embed_model,
        show_progress=True,
    )   
    uber_index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    print("Loading existing index...")
    # Load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    uber_index = load_index_from_storage(storage_context)

# Build query engine
print("Building query engine...")
uber_engine = uber_index.as_query_engine(llm=query_llm)

# Create tool
query_engine_tool = QueryEngineTool(
    query_engine=uber_engine,
    metadata=ToolMetadata(
        name="uber_10k",
        description=(
            f"Provides information about Uber financials for years {', '.join([str(year) for year in years])}. "
            "Use a detailed plain text question as input to the tool."
        ),
    ),
)

agent = FunctionCallingAgent.from_tools(
    [query_engine_tool, multiply_tool, add_tool], llm=query_llm, verbose=True
)

queries = [
    "Tell me 435345 times 234525.",
    "Tell me both the risk factors and tailwinds for Uber? Do two parallel tool calls.",
]

for query in queries:
    print(f"Query: {query}")

    response = agent.chat(query)

    print(f"Response: {str(response)}")
    print(f"Sources: {response.sources}")