# LlamaIndex with Amazon Bedrock

This directory demonstrates how to use LlamaIndex with Amazon Bedrock for building RAG applications and agents.

## Examples

- **01_rag.py**: Demonstrates Retrieval-Augmented Generation (RAG)
- **02_agent.py**: Shows how to build an agent using LlamaIndex and Amazon Bedrock

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have the necessary AWS credentials and permissions set up to access Bedrock services.

3. Run the initialization script to download example data:
   ```
   ./init_data.sh
   ```

4. Run the desired example script:
   ```
   python <script_name>.py
   ```

## Configuration

The examples use the Anthropic Claude 3 Sonnet model and Amazon Titan embeddings via Amazon Bedrock. The model IDs and region are specified in each script.
