# Example 4: GTP Packet Analysis with RAG

This example demonstrates how to use Retrieval-Augmented Generation (RAG) to analyze GTP (GPRS Tunneling Protocol) packets from PCAP files. The system can parse GTP packets, extract relevant information, and provide intelligent analysis using AI.

## Features

- **GTP Packet Parsing**: Extracts GTP header information, TEID, message types, and extension headers
- **ICMP Analysis**: Analyzes encapsulated ICMP packets within GTP tunnels
- **RAG Integration**: Uses Google Gemini AI for intelligent packet analysis
- **Vector Database**: Stores packet information for efficient retrieval and analysis
- **Comprehensive Reporting**: Generates detailed analysis reports

## Requirements

- Python 3.8+
- Google API Key for Gemini AI
- Required packages (see pyproject.toml)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Google API key:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

## Usage

1. **Basic GTP Analysis**:
```bash
python rag-gtp-analysis.py --pcap your_gtp_packets.pcap
```

2. **Interactive Analysis**:
```bash
python rag-gtp-analysis.py --pcap your_gtp_packets.pcap --interactive
```

3. **Generate Report**:
```bash
python rag-gtp-analysis.py --pcap your_gtp_packets.pcap --report
```

The `--report` flag generates a comprehensive analysis report (without source documents) and saves it to a file with the naming convention: `{PCAP_NAME}_RAG_{TIMESTAMP}.txt`

## Sample Output

The system analyzes GTP packets and provides insights such as:
- GTP tunnel information (TEID, version, protocol type)
- Encapsulated packet details (IP addresses, protocols)
- ICMP packet analysis (ping requests, responses)
- Network flow patterns
- Potential issues (no responses, timeouts)

## Supported GTP Features

- GTP-U (User Plane) packets
- T-PDU (Tunneled PDU) messages
- Extension headers (PDU Session Container)
- TEID (Tunnel Endpoint Identifier) analysis
- Encapsulated IP/ICMP analysis

## Example Questions

- "What GTP tunnels are active in this capture?"
- "Are there any ICMP packets without responses?"
- "What is the source and destination of the encapsulated traffic?"
- "Are there any unusual GTP message types?"
- "What is the TEID distribution in this capture?"
