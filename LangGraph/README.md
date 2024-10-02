# LangGraph with Amazon Bedrock

This directory demonstrates how to use LangGraph with Amazon Bedrock for building stateful, multi-actor applications with LLMs.

## Example

- **app.py**: Demonstrates building a graph-based application using LangGraph and Amazon Bedrock

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have the necessary AWS credentials and permissions set up to access Bedrock services.

3. Run the example script:
   ```
   python app.py
   ```

## Configuration

The example uses the Claude 3 Sonnet model via Amazon Bedrock. The model ID and region are specified in the script.