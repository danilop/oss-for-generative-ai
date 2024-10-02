import ell

import openai # openai v1.0.0+
client = openai.OpenAI(api_key="anything",base_url="http://0.0.0.0:8000") # set proxy to base_url


@ell.simple(model="gpt-3.5-turbo", client=client)
def hello(name: str):
    """You are a helpful assistant.""" # System prompt
    return f"Say hello to {name}!" # User prompt

greeting = hello("Sam Altman")
print(greeting)