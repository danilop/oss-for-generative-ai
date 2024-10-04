import dspy

# Set up the LM.
lm = dspy.LM("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")

dspy.settings.configure(lm=lm)

# Sentiment prediction

classify = dspy.Predict('sentence -> sentiment')

sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

response = classify(sentence=sentence)

print(response)

# You can access fields based on the signature

print(f"The sentiment is: {response.sentiment}")

#lm.inspect_history(n=1)

# Chain of Thought

document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought('document -> summary')

response = summarize(document=document)

print(response)

#lm.inspect_history(n=1)

# Chain of Tought with list of bool output

claims = ["Python was released in 1991.", "Python is a compiled language."]

fact_checking = dspy.ChainOfThought('claims -> verdicts: list[bool]')

response = fact_checking(claims=["Python was released in 1991.", "Python is a compiled language."])

print(response)

# Program of Thought (Python)

# Define a simple signature for basic question answering
generate_answer_signature = dspy.Signature("question -> answer")

# Pass signature to ProgramOfThought Module
pot = dspy.ProgramOfThought(generate_answer_signature)

# Call the ProgramOfThought module on a particular input
question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
response = pot(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ProgramOfThought process): {response.answer}")

# To see the PYthon code that was generated

lm.inspect_history(n=4)

print(response)
