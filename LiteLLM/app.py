import os
from litellm import completion

MODEL_ID = "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0"

messages = [{"role": "user", "content": "Tell me a nice story about Julius Caesar."}]

print("Invoking model")

response = completion(
  model=MODEL_ID,
  messages=messages,
)

print(response)

print("Content:")

print(response.choices[0].message.content)

print("Invoking model with streaming")

response = completion(
  model=MODEL_ID,
  messages=messages,
  stream=True,
)

for part in response:
    print(part.choices[0].delta.content or "", end='')
print()
