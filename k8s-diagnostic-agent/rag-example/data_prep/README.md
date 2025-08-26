# Data Preparation Module

This module processes PCAP files and generates vector embeddings for 5G network analysis using RAG (Retrieval-Augmented Generation).

## Overview

The data preparation module:
- Processes PCAP files to extract network features
- Generates embeddings using sentence transformers
- Stores results in a FAISS vector database
- Supports labeled data for supervised learning

## Prerequisites

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Configuration

Create a JSON configuration file (see `config-input.json` for example):
```json
{
    "input": [
        {
            "file": "your-pcap-file.pcap",
            "label": 0,
            "issue_type": "Success",
            "description": "Description of the traffic pattern",
            "key_patterns": ["pattern1", "pattern2"]
        }
    ]
}
```

### 2. Run Data Processing

Process PCAP files and generate embeddings:

**Option A: Run from rag-example directory (Recommended)**
```bash
cd k8s-diagnostic-agent/rag-example
python3 -m data_prep.process_pcaps --config data_prep/config-input.json --output data_prep/vector_store.faiss
```

**Option B: Run script directly (requires PYTHONPATH setup)**
```bash
cd k8s-diagnostic-agent/rag-example/data_prep
export PYTHONPATH="<base-dir-python-program-path>:$PYTHONPATH"
python3 process_pcaps.py --config config-input.json --output vector_store.faiss
```

**Arguments:**
- `--config`: Path to configuration JSON file (required)
- `--output`: Output path for FAISS index (default: vector_store.faiss)
- `--base-dir`: Base directory for PCAP file paths (optional)

### 3. Example Run

```bash
# From the rag-example directory
python3 -m data_prep.process_pcaps --config data_prep/config-input.json --output data_prep/vector_store.faiss --base-dir <base-dir-pcaps>
```

## Output

- **FAISS Index**: Vector database file containing embeddings
- **Console Output**: Processing summary with success/error counts
- **Metadata**: Stored with each embedding for retrieval

## Components

- **PCAPProcessor**: Extracts features from PCAP files
- **EmbeddingGenerator**: Creates vector embeddings using sentence transformers
- **VectorStore**: Manages FAISS vector database operations
- **Config**: Loads and validates configuration files

## Troubleshooting

- Ensure PCAP files exist and are accessible
- Check that all dependencies are installed
- Verify JSON configuration format is correct
- Monitor console output for error details
