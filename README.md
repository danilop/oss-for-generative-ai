# Open Source Frameworks for Building Generative AI Applications

This repository contains examples of popular open source frameworks for building generative AI applications and showcases how to use these frameworks with [Amazon Bedrock](https://aws.amazon.com/bedrock/).

## Frameworks Included

- **[LangChain](https://www.langchain.com/)**: A framework for developing applications powered by language models, featuring examples of:
  - Basic model invocation
  - Chaining prompts
  - Building an API
  - Creating a client
  - Implementing a chatbot
  - Using Bedrock Agents

- **[LangGraph](https://github.com/langchain-ai/langgraph)**: An extension of LangChain for building stateful, multi-actor applications with large language models (LLMs)

- **[Haystack](https://haystack.deepset.ai/)**: An end-to-end framework for building search systems and language model applications

- **[LlamaIndex](https://www.llamaindex.ai/)**: A data framework for LLM-based applications, with examples of:
  - RAG (Retrieval-Augmented Generation)
  - Building an agent

- **[DSPy](https://github.com/stanfordnlp/dspy)**: A framework for solving AI tasks using large language models

- **[RAGAS](https://github.com/explodinggradients/ragas)**: A framework for evaluating Retrieval Augmented Generation (RAG) pipelines

- **[LiteLLM](https://github.com/BerriAI/litellm)**: A library to standardize the use of LLMs from different providers

## Getting Started

Each framework has its own directory containing example scripts and a `requirements.txt` file listing the necessary dependencies.

To run the examples:

1. Clone this repository
2. Navigate to the desired framework's directory
3. For Python projects:
   - Create a virtual environment: `python -m venv .venv` (recommended)
   - Activate the virtual environment:
     - On Windows: `.venv\Scripts\activate`
     - On macOS and Linux: `source .venv/bin/activate`
   - Install the required dependencies: `pip install -r requirements.txt`
   - Run the example scripts

## Configuration

Ensure you have the necessary AWS credentials and permissions set up to access Amazon Bedrock.

## Frameworks Overview

### [LangChain](https://www.langchain.com/)
- A framework for developing applications powered by language models
- Key Features:
  - Modular components for LLM-powered applications
  - Chains and agents for complex LLM workflows
  - Memory systems for contextual interactions
  - Integration with various data sources and APIs
- Primary Use Cases:
  - Building conversational AI systems
  - Creating domain-specific question-answering systems
  - Developing AI-powered automation tools

### [LangGraph](https://github.com/langchain-ai/langgraph)
- An extension of LangChain for building stateful, multi-actor applications with LLMs
- Key Features:
  - Graph-based workflow management
  - State management for complex agent interactions
  - Tools for designing and implementing multi-agent systems
  - Cyclic workflows and feedback loops
- Primary Use Cases:
  - Creating collaborative AI agent systems
  - Implementing complex, stateful AI workflows
  - Developing AI-powered simulations and games

### [Haystack](https://haystack.deepset.ai/)
- An open-source framework for building production-ready LLM applications
- Key Features:
  - Composable AI systems with flexible pipelines
  - Multi-modal AI support (text, image, audio)
  - Production-ready with serializable pipelines and monitoring
- Primary Use Cases:
  - Building RAG pipelines and search systems
  - Developing conversational AI and chatbots
  - Content generation and summarization
  - Creating agentic pipelines with complex workflows

### [LlamaIndex](https://www.llamaindex.ai/)
- A data framework for building LLM-powered applications
- Key Features:
  - Advanced data ingestion and indexing
  - Query processing and response synthesis
  - Support for various data connectors
  - Customizable retrieval and ranking algorithms
- Primary Use Cases:
  - Creating knowledge bases and question-answering systems
  - Implementing semantic search over large datasets
  - Building context-aware AI assistants

### [DSPy](https://github.com/stanfordnlp/dspy)
- A framework for solving AI tasks through declarative and optimizable language model programs
- Key Features:
  - Declarative programming model for LLM interactions
  - Automatic optimization of LLM prompts and parameters
  - Signature-based type system for LLM inputs/outputs
  - Teleprompter (now optimizer) for automatic prompt improvement
- Primary Use Cases:
  - Developing robust and optimized NLP pipelines
  - Creating self-improving AI systems
  - Implementing complex reasoning tasks with LLMs

### [RAGAS](https://github.com/explodinggradients/ragas)
- An evaluation framework for Retrieval Augmented Generation (RAG) systems
- Key Features:
  - Automated evaluation of RAG pipelines
  - Multiple evaluation metrics (faithfulness, context relevancy, answer relevancy)
  - Support for different types of questions and datasets
  - Integration with popular RAG frameworks
- Primary Use Cases:
  - Benchmarking RAG system performance
  - Identifying areas for improvement in RAG pipelines
  - Comparing different RAG implementations

### [LiteLLM](https://github.com/BerriAI/litellm)
- A unified interface for multiple LLM providers
- Key Features:
  - Standardized API for 100+ LLM models
  - Automatic fallback and load balancing
  - Caching and retry mechanisms
  - Usage tracking and budget management
- Primary Use Cases:
  - Simplifying multi-LLM application development
  - Implementing model redundancy and fallback strategies
  - Managing LLM usage across different providers

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This is a collection of examples for educational purposes. Ensure you comply with the terms of service for Amazon Bedrock and any other services used when deploying these examples in production environments.