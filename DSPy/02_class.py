import dspy

# Set up the LM.
lm = dspy.LM("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")

dspy.settings.configure(lm=lm)

# Using a class to define the signature

class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context = dspy.InputField(desc="facts here are assumed to be true")
    text = dspy.InputField()
    faithfulness = dspy.OutputField(desc="True/False indicating if text is faithful to context")

context = "According to Roman tradition, the city of Rome was founded by twin brothers Romulus and Remus in 753 BCE. The twins were said to be the sons of Mars, the god of war, and were raised by a she-wolf. As adults, they decided to found a city, but disagreed on its location. Romulus wanted to build on the Palatine Hill, while Remus preferred the Aventine Hill. To settle their dispute, they agreed to consult augury, but this led to a quarrel in which Romulus killed Remus. Romulus then became the first king of Rome, which was named after him. The early Roman state was likely a kingdom, before transitioning to a republic around 509 BCE."

text = "Rome was founded by Romulus alone in 750 BCE."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
response = faithfulness(context=context, text=text)

print(response)

# Class + Chain of Thought with Hint

class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words", prefix="Question's Answer:")

# Pass signature to ChainOfThought module
generate_answer = dspy.ChainOfThoughtWithHint(BasicQA)

# Call the predictor on a particular input alongside a hint.
question='What is the color of the sky?'
hint = "It's what you often see during a sunny day."
pred = generate_answer(question=question, hint=hint)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")
