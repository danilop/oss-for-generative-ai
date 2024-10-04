import dspy

# Set up the LM.
lm = dspy.LM("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")

dspy.settings.configure(lm=lm)


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
