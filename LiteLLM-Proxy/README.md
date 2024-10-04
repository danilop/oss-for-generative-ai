# LiteLLM Proxy with Amazon Bedrock

**Currently not working***

This directory demonstrates how to set up and use a LiteLLM proxy server with Amazon Bedrock, allowing you to standardize API calls to different LLM providers.

## Examples

- **proxy.sh**: Shell script to start the LiteLLM proxy server
- **config.yaml**: Configuration file for the LiteLLM proxy
- **app.py**: Python script showing how to use the LiteLLM proxy with Amazon Bedrock

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have the necessary AWS credentials and permissions set up to access Bedrock services.

3. Start the LiteLLM proxy server:
   ```
   ./proxy.sh
   ```

4. In a separate terminal, run the example script:
   ```
   python app.py
   ```

## Configuration

The examples use the Claude 3.5 Sonnet model via Amazon Bedrock. The model ID and region are specified in the `config.yaml` file and the `proxy.sh` script.

## Usage

The LiteLLM proxy allows you to use a standardized API for multiple LLM providers. In this example, we're using it with Amazon Bedrock, but you can easily extend it to work with other providers by modifying the configuration.

The `app.py` script demonstrates how to use the LiteLLM proxy with the `ell` library, which provides a simple interface for working with language models.