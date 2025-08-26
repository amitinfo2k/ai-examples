# 5G Core PCAP Analysis Tool

A Proof of Concept (PoC) for analyzing 5G Core PCAP files to classify success/failure scenarios and provide diagnostic insights using Machine Learning and Retrieval-Augmented Generation (RAG).

## Features

- **PCAP Processing**: Extract features from 5G Core PCAP files
- **Machine Learning**: Classify PCAPs as success or failure
- **Explainability**: Provide human-readable explanations using RAG
- **Similarity Search**: Find similar historical PCAPs for comparison

## Project Structure

```
.
├── config.yaml              # Configuration file
├── pcap_analyzer/          # Main package
│   ├── __init__.py         # Package initialization
│   ├── cli.py              # Command-line interface
│   └── modules/
│       ├── data_prep/      # Data preparation module
│       ├── model_training/ # Model training module
│       └── testing/        # Testing and prediction module
├── data/                   # Data directories (created at runtime)
│   ├── raw_pcaps/          # Raw PCAP files
│   ├── processed/          # Processed features
│   └── embeddings/         # Vector embeddings
├── models/                 # Trained models
├── pyproject.toml         # Project configuration and dependencies
└── README.md              # This file
```

## Prerequisites

- Python 3.12.6
- `uv` package manager (alternative to pip)

## Virtual Environment Setup

1. First, install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create and activate a new virtual environment:
   ```bash
   # Create a new virtual environment in the project directory
   uv venv .venv
   
   # Activate the virtual environment
   # On Linux/macOS:
   source .venv/bin/activate
   # On Windows:
   # .venv\Scripts\activate
   ```

3. Upgrade pip and setuptools in the virtual environment:
   ```bash
   uv pip install --upgrade pip setuptools
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd 5g-pcap-analyzer
   ```

2. Install the package in development mode using `uv`:
   ```bash
   uv pip install -e .
   ```

   For production installation (without development dependencies):
   ```bash
   uv pip install .
   ```

## Usage

The tool provides three main commands:

### 1. Process PCAP Files

Extract features from PCAP files in a directory:

```bash
python -m pcap_analyzer.cli process data/raw_pcaps/ --output data/processed/features.json
```

### 2. Train the Model

Train a classification model using extracted features:

```bash
python -m pcap_analyzer.cli train data/processed/features.json --output-dir models/
```

### 3. Analyze a PCAP File

Classify a PCAP file and get an explanation:

```bash
python -m pcap_analyzer.cli predict data/raw_pcaps/sample_upf_success.pcap --model-dir models/ --output results/results.json
```

## Workflow

1. **Data Collection**: Gather PCAP files from your 5G Core network
2. **Feature Extraction**: Process PCAPs to extract relevant features
3. **Model Training**: Train a classifier on labeled data
4. **Prediction**: Classify new PCAPs and get explanations

## Configuration

Edit `config.yaml` to customize:
- Paths for data and models
- Feature extraction parameters
- Model hyperparameters
- RAG settings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Scapy, scikit-learn, and FAISS
- Uses Sentence Transformers for text embeddings
- Inspired by 5G network troubleshooting best practices
