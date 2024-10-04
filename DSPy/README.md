# DSPy with Amazon Bedrock

This directory demonstrates how to use DSPy with Amazon Bedrock for solving AI tasks using large language models.

## Example

- **01_signature.py**: Demonstrates using DSPy signatures with Amazon Bedrock
- **02_class.py**: Demonstrates using classes to define DSPy signatures
- **03_retriever.py**: Demonstrates using DSPy retriever
- **04_optimizer.py**: Demonstrates using DSPy optimizer (formerly teleprompter)

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have the necessary AWS credentials and permissions set up to access Bedrock services.

3. Run the example script:
   ```
   python <script name>.py
   ```

## Configuration

The example uses the Claude 3.5 Sonnet model via Amazon Bedrock. The model configuration is handled through DSPy's integration with Bedrock.