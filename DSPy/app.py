import dspy
import random

# Set up the LM.
lm = dspy.LM("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")

dspy.settings.configure(lm=lm)
dspy.settings.configure

sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.
classify = dspy.Predict('sentence -> sentiment')
response = classify(sentence=sentence)

print(response)

#lm.inspect_history(n=1)

document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)

print(response)

#lm.inspect_history(n=1)

fact_checking = dspy.ChainOfThought('claims -> verdicts: list[bool]')
response = fact_checking(claims=["Python was released in 1991.", "Python is a compiled language."])

print(response)

class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context = dspy.InputField(desc="facts here are assumed to be true")
    text = dspy.InputField()
    faithfulness = dspy.OutputField(desc="True/False indicating if text is faithful to context")

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

text = "Lee scored 3 goals for Colchester United."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
response = faithfulness(context=context, text=text)

print(response)


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words", prefix="Question's Answer:")

#Pass signature to ChainOfThought module
generate_answer = dspy.ChainOfThoughtWithHint(BasicQA)

# Call the predictor on a particular input alongside a hint.
question='What is the color of the sky?'
hint = "It's what you often see during a sunny day."
pred = generate_answer(question=question, hint=hint)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")



#Define a simple signature for basic question answering
generate_answer_signature = dspy.Signature("question -> answer")
#generate_answer_signature.attach(question=("Question:", "")).attach(answer=("Answer:", "often between 1 and 5 words"))

# Pass signature to ProgramOfThought Module
pot = dspy.ProgramOfThought(generate_answer_signature)

#Call the ProgramOfThought module on a particular input
question = 'Sarah has 5 apples. She buys 7 more apples from the store. How many apples does Sarah have now?'
response = pot(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ProgramOfThought process): {response.answer}")

# To see the PYthon code that was generated
#lm.inspect_history(n=4)

print(response)

# Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Pass signature to ReAct module
react_module = dspy.ReAct(BasicQA)

# Call the ReAct module on a particular input
question = 'Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?'
response = react_module(question=question)

print(f"Question: {question}")
print(f"Final Predicted Answer (after ReAct process): {response.answer}")


#Define a retrieval model server to send retrieval requests to
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

#Configure retrieval server internally
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

#Define Retrieve Module
retriever = dspy.Retrieve(k=3)

query='When was the first FIFA World Cup held?'

# Call the retriever on a particular query.
topK_passages = retriever(query).passages

print(f"Top {retriever.k} passages for question: {query} \n", '-' * 30, '\n')

for idx, passage in enumerate(topK_passages):
    print(f'{idx+1}]', passage, '\n')




from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Load math questions from the GSM8K dataset.
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset)

from dspy.evaluate import Evaluate

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)

lm.inspect_history(n=1)