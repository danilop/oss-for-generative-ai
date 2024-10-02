from datasets import Dataset 
import os
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, context_precision, context_utilization, answer_relevancy, answer_correctness
from langchain_aws import ChatBedrockConverse
from langchain_aws import BedrockEmbeddings

AWS_REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"

bedrock_model = ChatBedrockConverse(
    model=MODEL_ID,
    temperature=0,
    max_tokens=None,
    region_name=AWS_REGION,
)

bedrock_embeddings = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL_ID,
    region_name=AWS_REGION,
)

data_samples = {
    "question": [ # The user query that is the input of the RAG pipeline
        "When was Julius Caesar born?",
        "How did Julius Caesar die?",
        "What was Caesar's most famous military conquest?",
        "What political office did Caesar hold in 59 BCE?",
        "Who was Caesar's last wife?"
    ],
    "answer": [ # The generated answer from the RAG pipeline
        "Julius Caesar was born on July 12 or 13, 100 BCE",
        "Julius Caesar was assassinated on March 15, 44 BCE",
        "Caesar's conquest of Gaul (modern-day France and Belgium) was his most famous military achievement",
        "Caesar held the office of consul in 59 BCE",
        "Caesar's last wife was Cleopatra"  # This is incorrect
    ],
    "contexts": [ # The contexts retrieved from the external knowledge source used to answer the question
        ["Julius Caesar was born in Rome on 12 or 13 July 100 BC into the prestigious Julian clan.", 
         "He was born Gaius Julius Caesar, of the prestigious Julian clan."],
        ["On the Ides of March (15 March) of 44 BC, Caesar was assassinated by a group of rebellious senators.",
         "Caesar was stabbed 23 times and died at the base of Pompey's statue."],
        ["From 58 BC to 50 BC, Caesar led military campaigns in Gaul, extending Rome's territory to the English Channel and the Rhine.",
         "The Gallic Wars ended with complete Roman victory at the Battle of Alesia."],
        ["In Roman politics, the year 59 BCE was significant for Caesar's career advancement.",
         "Caesar formed a political alliance known as the First Triumvirate with Pompey and Crassus."],
        ["Julius Caesar was known to have had multiple marriages throughout his life.",
         "Marriage alliances were common among Roman politicians to forge political connections."]  # Generic context that doesn't provide the correct answer
    ],
    "ground_truth": [ # The ground truth answer to the question (the only human-annotated information)
        "Julius Caesar was born on July 12 or 13, 100 BCE in Rome",
        "Julius Caesar was assassinated on March 15, 44 BCE by a group of Roman senators",
        "Caesar's conquest of Gaul from 58 BC to 50 BC was his most famous military achievement",
        "Julius Caesar held the office of consul in 59 BCE",
        "Caesar's last wife was Calpurnia Pisonis"
    ]
}

dataset = Dataset.from_dict(data_samples)

result = evaluate(
    dataset,
    metrics=[faithfulness, context_recall, context_precision, context_utilization, answer_relevancy, answer_correctness],
    llm=bedrock_model,
    embeddings=bedrock_embeddings,
)

print("\nMetrics:")
print(result)
print("\nData:")
print(result.to_pandas().to_csv(index=False))