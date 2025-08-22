# PFCP RAG Analysis

> **⚠️ Development Status: In Progress**  
> This program is currently under development and may not provide the expected in-depth analysis. While it runs and processes PCAP files, the analysis capabilities are still being refined and enhanced.

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline for analyzing PFCP (Packet Forwarding Control Protocol) network logs using Python, LangChain, and the Google Gemini API. The system is designed to provide comprehensive 5G telecom protocol analysis, allowing you to ask natural language questions about your PFCP network traffic and get detailed, factual answers.

## Features

- **Advanced PFCP Protocol Analysis**: Deep parsing of PFCP messages including session establishment, modification, deletion, and association management
- **5G Protocol Support**: Comprehensive analysis of PFCP, GTP-U, S1AP, and NGAP protocols
- **Session Tracking**: Automatic matching of request/response pairs with success/failure analysis
- **Detailed Packet Inspection**: Full packet dissection with Information Element parsing
- **RAG-Powered Queries**: Natural language questions about network behavior and performance

## Prerequisites

You must have the following installed:

- Python 3.9+: This project requires Python version 3.9 or newer
- A Google API Key: You'll need an API key for the Gemini API. You can get one for free from the Google AI Studio website
- PCAP files: Network capture files containing PFCP traffic for analysis

## Installation

Install the required packages. With your virtual environment active, run the following command:

```bash
pip install langchain langchain-google-genai langchain-community chromadb pysqlite3-binary scapy
```

**Note**: This program requires Scapy with contrib modules for full PFCP protocol support.

## Usage

### 1. Set your Google API Key

The program needs your API key to access the Gemini models. Set it as an environment variable in your terminal session:

```bash
export GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

(Replace "YOUR_GEMINI_API_KEY" with your actual key.)

### 2. Run the program

Make sure you are in the directory containing `rag-tcpdump-pfcp-example.py` and run:

```bash
python3 rag-tcpdump-pfcp-example.py
```

### 3. Provide PCAP file

When prompted, enter the path to your `.pcap` file containing PFCP traffic.

### 4. Ask questions

The program will analyze your PFCP traffic and allow you to ask questions such as:
- "How many PFCP session establishment requests were successful?"
- "What is the success rate of session establishment?"
- "Show me details about failed PFCP associations"
- "How many heartbeat messages are in the capture?"

## How It Works

The script performs comprehensive analysis of your PCAP file:

1. **Packet Parsing**: Reads and parses each packet, identifying PFCP, GTP-U, S1AP, and NGAP protocols
2. **PFCP Analysis**: Deep analysis of PFCP messages including:
   - Message type identification (50+ different types)
   - Session Endpoint ID (SEID) tracking
   - Request/response pairing
   - Success/failure determination via Cause IE analysis
   - Information Element parsing
3. **Vector Indexing**: Converts parsed data into searchable vectors using Google's embedding model
4. **RAG Queries**: Uses Gemini 1.5 Flash to answer questions based on the indexed network data

## PFCP Protocol Support

This analyzer supports the complete PFCP protocol suite as defined in 3GPP TS 29.244:

- **Node Management**: Heartbeat, PFD Management, Association Setup/Update/Release
- **Session Management**: Establishment, Modification, Deletion, Reporting
- **Information Elements**: 80+ different IE types for comprehensive analysis
- **Success Analysis**: Automatic determination of request success via Cause codes

## Output Analysis

The program provides detailed statistics including:
- Total PFCP packet counts
- Session establishment success/failure rates
- Association management statistics
- Heartbeat message analysis
- Detailed packet-level information for troubleshooting

## Example Questions

- "What is the PFCP session establishment success rate?"
- "How many failed session modifications occurred?"
- "Show me all PFCP packets with SEID 12345"
- "What are the most common PFCP message types?"
- "Analyze the association setup process"

## Troubleshooting

- **Protocol Support**: Ensure Scapy contrib modules are properly loaded
- **PCAP Format**: Use standard PCAP/PCAPNG files with PFCP traffic
- **API Limits**: Be aware of Google Gemini API rate limits for large captures
- **Memory Usage**: Large PCAP files may require significant memory for processing

## Advanced Features

- **Multi-Protocol Analysis**: Simultaneous analysis of multiple 5G protocols
- **Real-time Processing**: Efficient handling of large network captures
- **Context-Aware Responses**: RAG system provides source packet information
- **Comprehensive Logging**: Detailed analysis logs for debugging and verification
