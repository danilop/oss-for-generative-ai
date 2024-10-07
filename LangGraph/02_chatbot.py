from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

AWS_REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"

model = ChatBedrockConverse(
    model=MODEL_ID,
    temperature=0,
    max_tokens=None,
    region_name=AWS_REGION,
)

print("Invoking model with single message")

output = model.invoke([HumanMessage(content="Hi! I'm Bob")])
output.pretty_print()

print("Invoking model with multiple messages (conversation)")

output = model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)
output.pretty_print()

print("Creating a graph")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define a new graph

# workflow = StateGraph(state_schema=MessagesState)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

workflow = StateGraph(state_schema=State)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Define the function that calls the model
# def call_model(state: MessagesState):
def call_model(state: State):
    chain = prompt | model
    #response = chain.invoke(state)
    trimmed_messages = trimmer.invoke(state["messages"])
    response = chain.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


### START - To draw the graph
app.get_graph().draw_mermaid_png(
    output_file_path="02_chatbot_graph.png",
)
### END - To draw the graph


config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."
language = "Italian"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()  # output contains all messages in state

query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "Spanish"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        for content in chunk.content:
            if 'type' in content and content['type'] == 'text':
                print(content['text'], end="|")

