# RAGAS with Amazon Bedrock

This directory demonstrates how to use RAGAS with Amazon Bedrock for evaluating Retrieval Augmented Generation (RAG) pipelines.

## Example

- **app.py**: Shows how to evaluate a RAG pipeline using RAGAS metrics and Amazon Bedrock

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

The example uses the Claude 3 Sonnet model for language tasks and Titan Embeddings for vector operations via Amazon Bedrock. The model IDs and region are specified in the script.