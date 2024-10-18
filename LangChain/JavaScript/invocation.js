import { ChatBedrockConverse } from "@langchain/aws";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";

const AWS_REGION = "us-east-1";
const MODEL_ID = "us.anthropic.claude-3-5-sonnet-20240620-v1:0";

// Creating the chat model using Amazon Bedrock Converse API

const chat_model = new ChatBedrockConverse({
  region: AWS_REGION,
  model: MODEL_ID,
  cache: true, // Memory cache
});

console.log("Using system and human message:")

const messages = [
  new SystemMessage("Translate the following from English into Italian"),
  new HumanMessage("hi!"),
];

const response = await chat_model.invoke(messages);

console.log(response.content);
console.log(response.response_metadata.usage);

console.log("Testing cache:")

const cached_response = await chat_model.invoke(messages);

console.log(cached_response.content);

console.log("Using streaming:")

const stream = await chat_model.stream([
  new HumanMessage("Tell me a joke about bears.")
]);

for await (const chunk of stream) {
  console.log('|' + chunk.content + '|'
  );
}

console.log("Using a longer conversation:")

const conversation_messages = [
  new SystemMessage("You are an expert US English copyeditor."),
  new HumanMessage("Is this sentence correct? 'I can't do nothing.'"),
];

const conversation_response = await chat_model.invoke(conversation_messages);

console.log(conversation_response.content);
console.log(conversation_response.response_metadata.usage);

console.log("Adding a reply:")

conversation_messages.push(conversation_response);
conversation_messages.push(new HumanMessage("Which one do you suggest?"))

const new_conversation_response = await chat_model.invoke(conversation_messages);

console.log(new_conversation_response.content);
console.log(new_conversation_response.response_metadata.usage);
