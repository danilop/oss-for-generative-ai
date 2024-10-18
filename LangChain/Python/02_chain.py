from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

AWS_REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"

model = ChatBedrockConverse(
    model=MODEL_ID,
    temperature=0,
    max_tokens=None,
    region_name=AWS_REGION,
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        (
            "human",
            "{input}",
        ),
    ]
)

parser = StrOutputParser()

chain = prompt_template | model | parser

output = chain.invoke(
    {
        "input_language": "English",
        "output_language": "German",
        "input": "I love programming.",
    }
)

print(output)
