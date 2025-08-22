# GTP Packet Analysis System - Usage Guide

This guide explains how to use the GTP (GPRS Tunneling Protocol) packet analysis system with RAG capabilities.

## Overview

The system consists of several components:
1. **Sample Packet Generator** - Creates test GTP packets
2. **GTP Analyzer** - Parses and analyzes GTP packets from PCAP files
3. **RAG System** - Uses Google Gemini AI for intelligent packet analysis
4. **Test Suite** - Validates the complete system

## Quick Start

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or install individually
pip install scapy langchain-google-genai langchain-community chromadb pysqlite3-binary
```

### 2. Set Google API Key

```bash
# Set your Google API key
export GOOGLE_API_KEY="your_api_key_here"

# Or set it in your shell profile
echo 'export GOOGLE_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Prepare Your PCAP File

Ensure you have a PCAP file containing GTP packets. The system will analyze:
- UDP packets on port 2152 (standard GTP port)
- GTP-U headers
- Encapsulated IP packets

### 4. Run the Analysis System

```bash
# Run the GTP analyzer with your PCAP file
python rag-gtp-analysis.py --pcap your_gtp_file.pcap

# Generate a comprehensive report
python rag-gtp-analysis.py --pcap your_gtp_file.pcap --report

# Or run the test suite
python test_gtp_system.py
```

## System Components

### Sample Packet Generator (`sample_gtp_packets.py`) - Optional

**Note:** This file is only needed for testing when you don't have real GTP PCAP files.

Creates test GTP packets including:
- GTP with ICMP Echo Request/Reply
- GTP with TCP data
- GTP Echo Request/Response
- Various TEIDs and message types

**Usage:**
```bash
python sample_gtp_packets.py
```

**Output:** `sample_gtp_packets.pcap`

### GTP Analyzer (`rag-gtp-analysis.py`)

Full-featured analysis system that:
- Parses PCAP files for GTP packets
- Extracts GTP header information with rich output
- Analyzes encapsulated protocols (ICMP, TCP, UDP)
- Provides comprehensive statistical summaries
- Supports interactive mode and detailed reporting

**Usage:**
```bash
# Basic analysis
python rag-gtp-analysis.py --pcap your_file.pcap

# Generate report
python rag-gtp-analysis.py --pcap your_file.pcap --report

# Interactive mode
python rag-gtp-analysis.py --pcap your_file.pcap --interactive
```

**Features:**
- Rich console output with tables and formatting
- TEID distribution analysis
- Message type counting with descriptions
- Encapsulated protocol identification
- ICMP packet analysis with patterns
- Extension header support
- Professional UI and comprehensive reporting



### Test Suite (`test_gtp_system.py`)

Validates the complete system:
- Dependency checking
- PCAP file availability
- Analysis system testing
- Interactive demo

**Usage:**
```bash
python test_gtp_system.py
```

### Report Generation

The GTP analyzer supports the `--report` flag to generate comprehensive analysis reports:

**Report Features:**
- Comprehensive GTP packet analysis
- AI-powered insights using RAG
- Clean, focused reports without source documents
- Structured question-answer format

**Report Naming Convention:**
```
{PCAP_FILENAME}_RAG_{TIMESTAMP}.txt
```

**Example:**
```
my_gtp_capture_RAG_20241201_143022.txt
```

**Usage:**
```bash
# Generate comprehensive report
python rag-gtp-analysis.py --pcap your_file.pcap --report
```

## Analyzing Your Own PCAP Files

### 1. Prepare Your PCAP File

Ensure your PCAP file contains GTP packets. The system looks for:
- UDP packets on port 2152 (standard GTP port)
- GTP-U headers
- Encapsulated IP packets

### 2. Run Analysis

```bash
# Replace with your PCAP file
python rag-gtp-analysis.py --pcap your_gtp_capture.pcap
```

### 3. Customize Analysis

Modify the analyzer classes to:
- Add new GTP message types
- Extract additional fields
- Analyze specific protocols
- Generate custom reports

## Understanding the Output

### Interactive Mode Behavior
- **Console Output**: Shows only questions and answers (no source documents)
- **Report Files**: Clean, focused reports without source documents
- **Clean Interface**: Focus on insights without cluttering output

### GTP Packet Information

Each GTP packet provides:
- **TEID**: Tunnel Endpoint Identifier (unique tunnel ID)
- **Message Type**: GTP message type (0xff = T-PDU, 0x01 = Echo Request, etc.)
- **Length**: Packet length
- **Version**: GTP version
- **Flags**: Extension headers, sequence numbers, etc.

### Encapsulated Protocol Analysis

The system identifies:
- **ICMP**: Echo requests/replies, error messages
- **TCP**: Connection attempts, data transfers
- **UDP**: DNS queries, other UDP traffic
- **IP**: Raw IP packets

### Statistical Analysis

- TEID distribution (how many packets per tunnel)
- Message type frequency
- Protocol breakdown
- Packet timing patterns

## Example Questions

The RAG system can answer questions like:

- "What GTP tunnels are active in this capture?"
- "Are there any ICMP packets without responses?"
- "What is the source and destination of the encapsulated traffic?"
- "Are there any unusual GTP message types?"
- "What is the TEID distribution in this capture?"
- "How many ICMP echo requests are there?"
- "What extension headers are present in the GTP packets?"
- "Are there any patterns in the packet timing?"

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Not Set**
   ```bash
   export GOOGLE_API_KEY="your_key_here"
   ```

3. **PCAP File Not Found**
   - Ensure the file path is correct
   - Check file permissions
   - Verify the file contains GTP packets

4. **GTP Protocol Not Supported**
   - Ensure Scapy is properly installed
   - Check if GTP contrib module is available

5. **Memory Issues with Large PCAP Files**
   - Use smaller capture files for testing
   - Implement packet filtering
   - Process files in chunks

### Debug Mode

Enable verbose output by modifying the scripts:
```python
# Add debug prints
print(f"Debug: Processing packet {packet_number}")
print(f"Debug: GTP layer found: {gtp_layer}")
```

## Extending the System

### Adding New GTP Message Types

```python
# In GTPAnalyzer class
GTP_MESSAGE_TYPES.update({
    0x50: "New Message Type",
    0x51: "Another Message Type"
})
```

### Adding New Protocol Analysis

```python
def _analyze_new_protocol(self, packet, gtp_info):
    """Analyze a new protocol type"""
    if NewProtocol in packet:
        new_protocol = packet[NewProtocol]
        gtp_info['new_protocol_field'] = new_protocol.field
```

### Custom RAG Prompts

```python
# Modify the prompt template
custom_prompt = """
You are a specialized GTP analyst. Focus on:
- Network performance issues
- Security anomalies
- Protocol compliance
- Traffic patterns

Context: {context}
Question: {question}
"""
```

## Integration Examples

### With Network Monitoring Tools

```python
# Integrate with existing monitoring
from your_monitoring_tool import PacketCapture

def analyze_gtp_stream(capture_stream):
    analyzer = GTPAnalyzer()
    for packet in capture_stream:
        analyzer._analyze_packet(packet)
    return analyzer.get_summary()
```

### With Logging Systems

```python
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add logging to analyzer
logger.info(f"Analyzed {len(self.gtp_packets)} GTP packets")
logger.warning(f"Found {len(self.icmp_analysis)} ICMP packets")
```

### With Web Interfaces

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_gtp():
    file = request.files['pcap']
    analyzer = GTPAnalyzer()
    results = analyzer.analyze_pcap(file)
    return jsonify(results)
```

## Performance Considerations

### Large PCAP Files

- Process files in chunks
- Implement packet filtering
- Use streaming analysis
- Consider database storage for results

### Memory Management

- Clear packet data after analysis
- Use generators for large datasets
- Implement cleanup methods
- Monitor memory usage

### API Rate Limits

- Implement request throttling
- Cache API responses
- Use batch processing
- Monitor API usage

## Security Considerations

- Never log sensitive packet data
- Sanitize output for external systems
- Implement access controls
- Audit analysis activities
- Secure API key storage

## Support and Contributing

### Getting Help

1. Check the troubleshooting section
2. Review example outputs
3. Test with sample data
4. Check dependency versions

### Contributing

1. Fork the repository
2. Create feature branches
3. Add tests for new functionality
4. Update documentation
5. Submit pull requests

### Reporting Issues

Include:
- Error messages
- PCAP file samples
- System information
- Steps to reproduce
- Expected vs actual behavior

## Conclusion

The GTP Packet Analysis System provides a powerful foundation for analyzing GTP traffic using AI-powered RAG capabilities. Start with the simplified version to understand the basics, then explore the advanced features as needed.

For production use, consider:
- Performance optimization
- Security hardening
- Integration with existing tools
- Custom analysis requirements
- Scalability planning
