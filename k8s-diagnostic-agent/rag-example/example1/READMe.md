# TCPdump RAG Analysis

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline for analyzing tcpdump network logs using Python, LangChain, and the Google Gemini API. The system allows you to ask natural language questions about your network traffic and get factual, grounded answers.

## Prerequisites

You must have the following installed:

- Python 3.9+: This project requires Python version 3.9 or newer.
- A Google API Key: You'll need an API key for the Gemini API. You can get one for free from the Google AI Studio website.

## Installation

Install the required packages.

With your virtual environment active, run the following command to install all the necessary libraries:

```bash
pip install langchain langchain-google-genai langchain-community chromadb pysqlite3-binary
```

## Usage

Set your Google API Key.

The program needs your API key to access the Gemini models. Set it as an environment variable in your terminal session.

```bash
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

(Replace "YOUR_GEMINI_API_KEY" with your actual key.)

Run the program.

Make sure you are in the directory containing rag-tcpdump-example.py and run the following command with your virtual environment active:

```bash
python3 rag-tcpdump-example.py
```

The program will then:

- Parse and index the sample tcpdump data.
- Run a series of pre-defined questions against the RAG system.
- Print the generated answers and the retrieved source documents used to create them.

## How It Works

The script first ingests the raw tcpdump logs, splits them into manageable chunks, and converts them into numerical vectors (embeddings) using the `models/embedding-001` model. These embeddings are stored in a local vector database (ChromaDB).

When a question is asked, the system retrieves the most relevant log entries from the database, augments a prompt with this context, and sends it to the `gemini-1.5-flash` model. The LLM then generates a factual answer based on the provided information, effectively "reasoning" over your private data.

For more understanding of the code flow, please follow the [Developer Guide](developerGuide.md). 