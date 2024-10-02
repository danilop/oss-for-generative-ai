# LiteLLM with Amazon Bedrock

This directory demonstrates how to use LiteLLM with Amazon Bedrock to standardize the use of LLMs.

## Examples

- **app.py**: Shows basic usage of LiteLLM with Amazon Bedrock
- **LiteLLM-Proxy/**: Contains examples of using LiteLLM as a proxy server

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

4. To run the proxy server (in the LiteLLM-Proxy directory):
   ```
   ./proxy.sh
   ```

## Configuration

The examples use the Claude 3 Sonnet model via Amazon Bedrock. The model ID is specified in each script or configuration file.