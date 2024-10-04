# LangGraph with Amazon Bedrock

This directory demonstrates how to use LangGraph with Amazon Bedrock for building stateful, multi-actor applications with LLMs.

## Example

- **01_tools.py**: Demonstrates building a graph-based application with tools using LangGraph and Amazon Bedrock
- **02_chatbot.py**: Demonstrates building a graph-based chatbot using LangGraph and Amazon Bedrock

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have the necessary AWS credentials and permissions set up to access Bedrock services.

3. Run the example script:
   ```
   python <script_name>.py
   ```

## Configuration

The example uses the Claude 3.5 Sonnet model via Amazon Bedrock. The model ID and region are specified in the script.