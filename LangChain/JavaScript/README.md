# LangChain.js with Amazon Bedrock

## Using Node

```
npm install
node invocation.js
```

## Using Deno

```
deno add npm:@langchain/aws
deno invocation.js
```

## Using Deno to compile into a self contained executable

```
deno compile --allow-read --allow-env --allow-net --allow-sys invocation.js
./invocation
```
