#!/usr/bin/env python3
"""
GTP Packet Analysis using RAG with Google Gemini AI

This script analyzes GTP (GPRS Tunneling Protocol) packets from PCAP files
and provides intelligent analysis using Retrieval-Augmented Generation.
"""

# Fix for sqlite3 on Linux
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import argparse
import getpass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter

# Scapy imports for packet analysis
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw
from scapy.all import load_contrib

# LangChain imports for RAG
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from tabulate import tabulate

# Initialize console for rich output
console = Console()

class GTPAnalyzer:
    """
    Enhanced GTP protocol analyzer for in-depth telecom protocol analysis
    """
    
    # GTP Message Types with detailed descriptions
    GTP_MESSAGE_TYPES = {
        0xff: "T-PDU (Tunneled PDU)",
        0x01: "Echo Request",
        0x02: "Echo Response",
        0x26: "Error Indication",
        0x31: "Supported Extension Headers Notification",
        0x32: "End Marker",
        0x34: "End Marker PDU Session Container",
        0x35: "CP PDU Session Information",
        0x36: "DL PDU Session Information",
        0x37: "UL PDU Session Information",
        0x38: "DL PDU Session Information (QoS Monitoring)",
        0x39: "UL PDU Session Information (QoS Monitoring)",
        0x3a: "DL PDU Session Information (Delay Indication)",
        0x3b: "UL PDU Session Information (Delay Indication)",
        0x3c: "DL PDU Session Information (N3/N9 Delay Indication)",
        0x3d: "UL PDU Session Information (N3/N9 Delay Indication)",
        0x3e: "DL PDU Session Information (Sequence Number)",
        0x3f: "UL PDU Session Information (Sequence Number)",
        0x40: "DL PDU Session Information (New IE Flag)",
        0x41: "UL PDU Session Information (New IE Flag)"
    }
    
    # Extension Header Types
    EXTENSION_HEADER_TYPES = {
        0x00: "No more extension headers",
        0x85: "PDU Session Container",
        0x01: "Reserved",
        0x02: "Reserved",
        0x03: "Reserved",
        0x04: "Reserved",
        0x05: "Reserved",
        0x06: "Reserved",
        0x07: "Reserved",
        0x08: "Reserved",
        0x09: "Reserved",
        0x0a: "Reserved",
        0x0b: "Reserved",
        0x0c: "Reserved",
        0x0d: "Reserved",
        0x0e: "Reserved",
        0x0f: "Reserved",
        0x10: "Reserved",
        0x11: "Reserved",
        0x12: "Reserved",
        0x13: "Reserved",
        0x14: "Reserved",
        0x15: "Reserved",
        0x16: "Reserved",
        0x17: "Reserved",
        0x18: "Reserved",
        0x19: "Reserved",
        0x1a: "Reserved",
        0x1b: "Reserved",
        0x1c: "Reserved",
        0x1d: "Reserved",
        0x1e: "Reserved",
        0x1f: "Reserved",
        0x20: "Reserved",
        0x21: "Reserved",
        0x22: "Reserved",
        0x23: "Reserved",
        0x24: "Reserved",
        0x25: "Reserved",
        0x26: "Reserved",
        0x27: "Reserved",
        0x28: "Reserved",
        0x29: "Reserved",
        0x2a: "Reserved",
        0x2b: "Reserved",
        0x2c: "Reserved",
        0x2d: "Reserved",
        0x2e: "Reserved",
        0x2f: "Reserved",
        0x30: "Reserved",
        0x31: "Reserved",
        0x32: "Reserved",
        0x33: "Reserved",
        0x34: "Reserved",
        0x35: "Reserved",
        0x36: "Reserved",
        0x37: "Reserved",
        0x38: "Reserved",
        0x39: "Reserved",
        0x3a: "Reserved",
        0x3b: "Reserved",
        0x3c: "Reserved",
        0x3d: "Reserved",
        0x3e: "Reserved",
        0x3f: "Reserved",
        0x40: "Reserved",
        0x41: "Reserved",
        0x42: "Reserved",
        0x43: "Reserved",
        0x44: "Reserved",
        0x45: "Reserved",
        0x46: "Reserved",
        0x47: "Reserved",
        0x48: "Reserved",
        0x49: "Reserved",
        0x4a: "Reserved",
        0x4b: "Reserved",
        0x4c: "Reserved",
        0x4d: "Reserved",
        0x4e: "Reserved",
        0x4f: "Reserved",
        0x50: "Reserved",
        0x51: "Reserved",
        0x52: "Reserved",
        0x53: "Reserved",
        0x54: "Reserved",
        0x55: "Reserved",
        0x56: "Reserved",
        0x57: "Reserved",
        0x58: "Reserved",
        0x59: "Reserved",
        0x5a: "Reserved",
        0x5b: "Reserved",
        0x5c: "Reserved",
        0x5d: "Reserved",
        0x5e: "Reserved",
        0x5f: "Reserved",
        0x60: "Reserved",
        0x61: "Reserved",
        0x62: "Reserved",
        0x63: "Reserved",
        0x64: "Reserved",
        0x65: "Reserved",
        0x66: "Reserved",
        0x67: "Reserved",
        0x68: "Reserved",
        0x69: "Reserved",
        0x6a: "Reserved",
        0x6b: "Reserved",
        0x6c: "Reserved",
        0x6d: "Reserved",
        0x6e: "Reserved",
        0x6f: "Reserved",
        0x70: "Reserved",
        0x71: "Reserved",
        0x72: "Reserved",
        0x73: "Reserved",
        0x74: "Reserved",
        0x75: "Reserved",
        0x76: "Reserved",
        0x77: "Reserved",
        0x78: "Reserved",
        0x79: "Reserved",
        0x7a: "Reserved",
        0x7b: "Reserved",
        0x7c: "Reserved",
        0x7d: "Reserved",
        0x7e: "Reserved",
        0x7f: "Reserved",
        0x80: "Reserved",
        0x81: "Reserved",
        0x82: "Reserved",
        0x83: "Reserved",
        0x84: "Reserved",
        0x86: "Reserved",
        0x87: "Reserved",
        0x88: "Reserved",
        0x89: "Reserved",
        0x8a: "Reserved",
        0x8b: "Reserved",
        0x8c: "Reserved",
        0x8d: "Reserved",
        0x8e: "Reserved",
        0x8f: "Reserved",
        0x90: "Reserved",
        0x91: "Reserved",
        0x92: "Reserved",
        0x93: "Reserved",
        0x94: "Reserved",
        0x95: "Reserved",
        0x96: "Reserved",
        0x97: "Reserved",
        0x98: "Reserved",
        0x99: "Reserved",
        0x9a: "Reserved",
        0x9b: "Reserved",
        0x9c: "Reserved",
        0x9d: "Reserved",
        0x9e: "Reserved",
        0x9f: "Reserved",
        0xa0: "Reserved",
        0xa1: "Reserved",
        0xa2: "Reserved",
        0xa3: "Reserved",
        0xa4: "Reserved",
        0xa5: "Reserved",
        0xa6: "Reserved",
        0xa7: "Reserved",
        0xa8: "Reserved",
        0xa9: "Reserved",
        0xaa: "Reserved",
        0xab: "Reserved",
        0xac: "Reserved",
        0xad: "Reserved",
        0xae: "Reserved",
        0xaf: "Reserved",
        0xb0: "Reserved",
        0xb1: "Reserved",
        0xb2: "Reserved",
        0xb3: "Reserved",
        0xb4: "Reserved",
        0xb5: "Reserved",
        0xb6: "Reserved",
        0xb7: "Reserved",
        0xb8: "Reserved",
        0xb9: "Reserved",
        0xba: "Reserved",
        0xbb: "Reserved",
        0xbc: "Reserved",
        0xbd: "Reserved",
        0xbe: "Reserved",
        0xbf: "Reserved",
        0xc0: "Reserved",
        0xc1: "Reserved",
        0xc2: "Reserved",
        0xc3: "Reserved",
        0xc4: "Reserved",
        0xc5: "Reserved",
        0xc6: "Reserved",
        0xc7: "Reserved",
        0xc8: "Reserved",
        0xc9: "Reserved",
        0xca: "Reserved",
        0xcb: "Reserved",
        0xcc: "Reserved",
        0xcd: "Reserved",
        0xce: "Reserved",
        0xcf: "Reserved",
        0xd0: "Reserved",
        0xd1: "Reserved",
        0xd2: "Reserved",
        0xd3: "Reserved",
        0xd4: "Reserved",
        0xd5: "Reserved",
        0xd6: "Reserved",
        0xd7: "Reserved",
        0xd8: "Reserved",
        0xd9: "Reserved",
        0xda: "Reserved",
        0xdb: "Reserved",
        0xdc: "Reserved",
        0xdd: "Reserved",
        0xde: "Reserved",
        0xdf: "Reserved",
        0xe0: "Reserved",
        0xe1: "Reserved",
        0xe2: "Reserved",
        0xe3: "Reserved",
        0xe4: "Reserved",
        0xe5: "Reserved",
        0xe6: "Reserved",
        0xe7: "Reserved",
        0xe8: "Reserved",
        0xe9: "Reserved",
        0xea: "Reserved",
        0xeb: "Reserved",
        0xec: "Reserved",
        0xed: "Reserved",
        0xee: "Reserved",
        0xef: "Reserved",
        0xf0: "Reserved",
        0xf1: "Reserved",
        0xf2: "Reserved",
        0xf3: "Reserved",
        0xf4: "Reserved",
        0xf5: "Reserved",
        0xf6: "Reserved",
        0xf7: "Reserved",
        0xf8: "Reserved",
        0xf9: "Reserved",
        0xfa: "Reserved",
        0xfb: "Reserved",
        0xfc: "Reserved",
        0xfd: "Reserved",
        0xfe: "Reserved"
    }
    
    def __init__(self):
        """Initialize the GTP analyzer"""
        self.gtp_packets = []
        self.analysis_summary = {}
        self.teid_distribution = Counter()
        self.message_type_distribution = Counter()
        self.encapsulated_protocols = Counter()
        self.icmp_analysis = []
        
    def analyze_pcap(self, pcap_file: str) -> Dict[str, Any]:
        """
        Analyze GTP packets from a PCAP file
        
        Args:
            pcap_file: Path to the PCAP file
            
        Returns:
            Dictionary containing analysis results
        """
        console.print(f"[bold blue]Analyzing PCAP file: {pcap_file}[/bold blue]")
        
        try:
            # Load GTP protocol support
            load_contrib("gtp")
            from scapy.contrib.gtp import GTP_U_Header
            
            # Read PCAP file
            packets = rdpcap(pcap_file)
            console.print(f"[green]Loaded {len(packets)} packets from PCAP file[/green]")
            
            # Analyze each packet
            for i, packet in enumerate(packets):
                self._analyze_packet(packet, i + 1)
            
            # Generate summary
            self._generate_summary()
            
            return self.analysis_summary
            
        except Exception as e:
            console.print(f"[red]Error analyzing PCAP file: {e}[/red]")
            return {}
    
    def _analyze_packet(self, packet, packet_number: int):
        """Analyze individual packet for GTP content"""
        try:
            # Check if packet has GTP layer
            if GTP_U_Header in packet:
                gtp_info = self._extract_gtp_info(packet, packet_number)
                if gtp_info:
                    self.gtp_packets.append(gtp_info)
                    
                    # Update counters
                    self.teid_distribution[gtp_info['teid']] += 1
                    self.message_type_distribution[gtp_info['message_type']] += 1
                    
                    # Analyze encapsulated content
                    if 'encapsulated_protocol' in gtp_info:
                        self.encapsulated_protocols[gtp_info['encapsulated_protocol']] += 1
                        
                        # Special analysis for ICMP
                        if gtp_info['encapsulated_protocol'] == 'ICMP':
                            self._analyze_icmp_packet(packet, gtp_info)
                            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not analyze packet {packet_number}: {e}[/yellow]")
    
    def _extract_gtp_info(self, packet, packet_number: int) -> Optional[Dict[str, Any]]:
        """Extract GTP information from packet"""
        try:
            gtp_layer = packet[GTP_U_Header]
            
            # Basic GTP header info
            gtp_info = {
                'packet_number': packet_number,
                'timestamp': packet.time,
                'version': gtp_layer.version,
                'protocol_type': gtp_layer.proto,
                'reserved': gtp_layer.reserved,
                'extension_header': gtp_layer.E,
                'sequence_number': gtp_layer.S,
                'npdu_number': gtp_layer.PN,
                'message_type': gtp_layer.gtp_type,
                'length': gtp_layer.length,
                'teid': gtp_layer.teid,
                'raw_data': bytes(gtp_layer.payload) if gtp_layer.payload else b''
            }
            
            # Add message type description
            gtp_info['message_type_desc'] = self.GTP_MESSAGE_TYPES.get(
                gtp_info['message_type'], 'Unknown'
            )
            
            # Analyze extension headers if present
            if gtp_info['extension_header']:
                gtp_info['extension_headers'] = self._analyze_extension_headers(gtp_layer)
            
            # Analyze encapsulated content
            gtp_info.update(self._analyze_encapsulated_content(packet))
            
            return gtp_info
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not extract GTP info: {e}[/yellow]")
            return None
    
    def _analyze_extension_headers(self, gtp_layer) -> List[Dict[str, Any]]:
        """Analyze GTP extension headers"""
        extension_headers = []
        
        try:
            # This is a simplified analysis - in practice, you'd need to parse
            # the extension header data more carefully
            if hasattr(gtp_layer, 'next_ext_hdr'):
                ext_hdr_type = gtp_layer.next_ext_hdr
                ext_hdr_desc = self.EXTENSION_HEADER_TYPES.get(ext_hdr_type, 'Unknown')
                
                extension_headers.append({
                    'type': ext_hdr_type,
                    'description': ext_hdr_desc
                })
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not analyze extension headers: {e}[/yellow]")
        
        return extension_headers
    
    def _analyze_encapsulated_content(self, packet) -> Dict[str, Any]:
        """Analyze the content encapsulated within GTP"""
        encapsulated_info = {}
        
        try:
            # Look for IP layer after GTP
            if IP in packet:
                ip_layer = packet[IP]
                encapsulated_info['encapsulated_protocol'] = 'IP'
                encapsulated_info['src_ip'] = ip_layer.src
                encapsulated_info['dst_ip'] = ip_layer.dst
                encapsulated_info['ip_protocol'] = ip_layer.proto
                
                # Check for ICMP
                if ICMP in packet:
                    icmp_layer = packet[ICMP]
                    encapsulated_info['encapsulated_protocol'] = 'ICMP'
                    encapsulated_info['icmp_type'] = icmp_layer.type
                    encapsulated_info['icmp_code'] = icmp_layer.code
                    encapsulated_info['icmp_id'] = icmp_layer.id
                    encapsulated_info['icmp_seq'] = icmp_layer.seq
                    
                # Check for TCP
                elif TCP in packet:
                    tcp_layer = packet[TCP]
                    encapsulated_info['encapsulated_protocol'] = 'TCP'
                    encapsulated_info['src_port'] = tcp_layer.sport
                    encapsulated_info['dst_port'] = tcp_layer.dport
                    
                # Check for UDP
                elif UDP in packet:
                    udp_layer = packet[UDP]
                    encapsulated_info['encapsulated_protocol'] = 'UDP'
                    encapsulated_info['src_port'] = udp_layer.sport
                    encapsulated_info['dst_port'] = udp_layer.dport
                    
        except Exception as e:
            console.print(f"[yellow]Warning: Could not analyze encapsulated content: {e}[/yellow]")
        
        return encapsulated_info
    
    def _analyze_icmp_packet(self, packet, gtp_info: Dict[str, Any]):
        """Special analysis for ICMP packets"""
        try:
            if ICMP in packet:
                icmp_layer = packet[ICMP]
                
                icmp_analysis = {
                    'packet_number': gtp_info['packet_number'],
                    'teid': gtp_info['teid'],
                    'timestamp': gtp_info['timestamp'],
                    'icmp_type': icmp_layer.type,
                    'icmp_code': icmp_layer.code,
                    'icmp_id': icmp_layer.id,
                    'icmp_seq': icmp_layer.seq,
                    'src_ip': gtp_info.get('src_ip'),
                    'dst_ip': gtp_info.get('dst_ip'),
                    'is_echo_request': icmp_layer.type == 8,
                    'is_echo_reply': icmp_layer.type == 0
                }
                
                self.icmp_analysis.append(icmp_analysis)
                
        except Exception as e:
            console.print(f"[yellow]Warning: Could not analyze ICMP packet: {e}[/yellow]")
    
    def _generate_summary(self):
        """Generate analysis summary"""
        self.analysis_summary = {
            'total_packets': len(self.gtp_packets),
            'unique_teids': len(self.teid_distribution),
            'message_types': dict(self.message_type_distribution),
            'encapsulated_protocols': dict(self.encapsulated_protocols),
            'icmp_packets': len(self.icmp_analysis),
            'teid_distribution': dict(self.teid_distribution),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_gtp_packets_text(self) -> List[str]:
        """Convert GTP packet analysis to text for RAG"""
        texts = []
        
        for packet in self.gtp_packets:
            text = f"""
Packet {packet['packet_number']}:
- Timestamp: {packet['timestamp']}
- GTP Version: {packet['version']}
- Message Type: {packet['message_type']} ({packet['message_type_desc']})
- TEID: {packet['teid']}
- Length: {packet['length']}
- Extension Header: {packet['extension_header']}
- Sequence Number: {packet['sequence_number']}
- N-PDU Number: {packet['npdu_number']}
"""
            
            if 'encapsulated_protocol' in packet:
                text += f"- Encapsulated Protocol: {packet['encapsulated_protocol']}\n"
                
                if packet['encapsulated_protocol'] == 'IP':
                    text += f"  - Source IP: {packet.get('src_ip', 'N/A')}\n"
                    text += f"  - Destination IP: {packet.get('dst_ip', 'N/A')}\n"
                    
                if packet['encapsulated_protocol'] == 'ICMP':
                    text += f"  - ICMP Type: {packet.get('icmp_type', 'N/A')}\n"
                    text += f"  - ICMP Code: {packet.get('icmp_code', 'N/A')}\n"
                    text += f"  - ICMP ID: {packet.get('icmp_id', 'N/A')}\n"
                    text += f"  - ICMP Sequence: {packet.get('icmp_seq', 'N/A')}\n"
            
            if 'extension_headers' in packet and packet['extension_headers']:
                text += "- Extension Headers:\n"
                for ext_hdr in packet['extension_headers']:
                    text += f"  - Type: {ext_hdr['type']} ({ext_hdr['description']})\n"
            
            texts.append(text)
        
        return texts
    
    def print_summary(self):
        """Print analysis summary to console"""
        console.print("\n[bold cyan]GTP Packet Analysis Summary[/bold cyan]")
        console.print("=" * 50)
        
        # Basic statistics
        table = Table(title="Packet Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total GTP Packets", str(self.analysis_summary['total_packets']))
        table.add_row("Unique TEIDs", str(self.analysis_summary['unique_teids']))
        table.add_row("ICMP Packets", str(self.analysis_summary['icmp_packets']))
        
        console.print(table)
        
        # Message type distribution
        if self.message_type_distribution:
            console.print("\n[bold yellow]Message Type Distribution:[/bold yellow]")
            msg_table = Table()
            msg_table.add_column("Message Type", style="cyan")
            msg_table.add_column("Count", style="green")
            msg_table.add_column("Description", style="yellow")
            
            for msg_type, count in self.message_type_distribution.most_common():
                desc = self.GTP_MESSAGE_TYPES.get(msg_type, 'Unknown')
                msg_table.add_row(f"0x{msg_type:02x}", str(count), desc)
            
            console.print(msg_table)
        
        # TEID distribution
        if self.teid_distribution:
            console.print("\n[bold yellow]Top TEIDs by Packet Count:[/bold yellow]")
            teid_table = Table()
            teid_table.add_column("TEID", style="cyan")
            teid_table.add_column("Packet Count", style="green")
            
            for teid, count in self.teid_distribution.most_common(10):
                teid_table.add_row(f"0x{teid:08x}", str(count))
            
            console.print(teid_table)
        
        # ICMP analysis
        if self.icmp_analysis:
            console.print("\n[bold yellow]ICMP Packet Analysis:[/bold yellow]")
            icmp_table = Table()
            icmp_table.add_column("Packet", style="cyan")
            icmp_table.add_column("TEID", style="green")
            icmp_table.add_column("Type", style="yellow")
            icmp_table.add_column("ID", style="magenta")
            icmp_table.add_column("Seq", style="blue")
            icmp_table.add_column("Source IP", style="cyan")
            icmp_table.add_column("Dest IP", style="cyan")
            
            for icmp in self.icmp_analysis:
                icmp_type_desc = "Echo Request" if icmp['is_echo_request'] else "Echo Reply" if icmp['is_echo_reply'] else f"Type {icmp['icmp_type']}"
                icmp_table.add_row(
                    str(icmp['packet_number']),
                    f"0x{icmp['teid']:08x}",
                    icmp_type_desc,
                    str(icmp['icmp_id']),
                    str(icmp['icmp_seq']),
                    icmp['src_ip'] or 'N/A',
                    icmp['dst_ip'] or 'N/A'
                )
            
            console.print(icmp_table)


class GTPRAGSystem:
    """
    RAG system for GTP packet analysis using Google Gemini AI
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the RAG system
        
        Args:
            api_key: Google API key for Gemini AI
        """
        self.api_key = api_key
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        
        # Set environment variable
        os.environ["GOOGLE_API_KEY"] = api_key
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models and vector store"""
        try:
            console.print("[green]Initializing Google Gemini AI models...[/green]")
            
            # Initialize embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Initialize LLM
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            
            console.print("[green]✓ Models initialized successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]Error initializing models: {e}[/red]")
            raise
    
    def create_vectorstore(self, gtp_analyzer: GTPAnalyzer):
        """
        Create vector store from GTP packet analysis
        
        Args:
            gtp_analyzer: GTPAnalyzer instance with packet data
        """
        try:
            console.print("[green]Creating vector store from GTP packet data...[/green]")
            
            # Get text representations of packets
            packet_texts = gtp_analyzer.get_gtp_packets_text()
            
            if not packet_texts:
                console.print("[yellow]No GTP packet data to process[/yellow]")
                return
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            # Split texts into chunks
            texts = text_splitter.create_documents(packet_texts)
            console.print(f"[green]Created {len(texts)} text chunks[/green]")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings
            )
            
            console.print("[green]✓ Vector store created successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]Error creating vector store: {e}[/red]")
            raise
    
    def setup_qa_chain(self):
        """Setup the question-answering chain"""
        try:
            console.print("[green]Setting up QA chain...[/green]")
            
            # Create prompt template
            rag_prompt_template = """
You are a GTP (GPRS Tunneling Protocol) packet analysis expert. Your role is to analyze GTP packet data and provide intelligent insights based on the provided context.

Context:
{context}

Question:
{question}

Instructions:
1. Answer based ONLY on the provided GTP packet context
2. If the answer is not in the context, say "I could not find the answer in the provided GTP packet data"
3. Provide comprehensive analysis covering ALL packets in the context, not just a subset
4. Use technical terminology appropriate for GTP protocol analysis
5. If analyzing ICMP packets, note any patterns or issues across ALL packets
6. For TEID analysis, provide hexadecimal values and packet counts for ALL TEIDs
7. When counting packets, ensure you analyze the complete dataset provided
8. Provide statistical summaries that reflect the total number of packets analyzed

Answer:
"""
            
            rag_prompt = PromptTemplate.from_template(rag_prompt_template)
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 100}),  # Increased to capture all packets
                chain_type_kwargs={"prompt": rag_prompt},
                return_source_documents=True
            )
            
            console.print("[green]✓ QA chain setup successfully[/green]")
            
        except Exception as e:
            console.print(f"[red]Error setting up QA chain: {e}[/red]")
            raise
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question about the GTP packet data
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")
        
        try:
            console.print(f"\n[bold blue]Question: {question}[/bold blue]")
            
            response = self.qa_chain.invoke({"query": question})
            
            console.print("[bold green]Answer:[/bold green]")
            console.print(response["result"])
            
            return response
            
        except Exception as e:
            console.print(f"[red]Error asking question: {e}[/red]")
            return {"error": str(e)}
    
    def generate_report(self, pcap_filename: str) -> str:
        """Generate comprehensive analysis report and save to file"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")
        
        # Create filename with PCAP name, timestamp, and RAG
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pcap_basename = os.path.splitext(os.path.basename(pcap_filename))[0]
        report_filename = f"{pcap_basename}_RAG_{timestamp}.txt"
        
        console.print(f"\n[bold blue]Generating comprehensive report: {report_filename}[/bold blue]")
        
        # Verify packet count for debugging
        if hasattr(self, 'vectorstore') and self.vectorstore:
            try:
                # Get total documents in vector store
                total_docs = len(self.vectorstore.get()['documents'])
                console.print(f"[yellow]Debug: Vector store contains {total_docs} documents[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Debug: Could not verify vector store size: {e}[/yellow]")
        
        # Define report questions
        report_questions = [
            "What GTP tunnels are active in this capture?",
            "Are there any ICMP packets without responses?",
            "What is the source and destination of the encapsulated traffic?",
            "Are there any unusual GTP message types?",
            "What is the TEID distribution in this capture?",
            "How many ICMP echo requests are there?",
            "What extension headers are present in the GTP packets?",
            "Are there any patterns in the packet timing?",
            "What is the overall health of the GTP tunnels?",
            "Are there any potential issues or anomalies?"
        ]
        
        # Generate report content
        report_content = []
        report_content.append("=" * 80)
        report_content.append("GTP PACKET ANALYSIS REPORT")
        report_content.append("=" * 80)
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"PCAP File: {pcap_filename}")
        report_content.append(f"Report File: {report_filename}")
        report_content.append("=" * 80)
        report_content.append("")
        
        # Add analysis questions and answers
        for i, question in enumerate(report_questions, 1):
            report_content.append(f"Question {i}: {question}")
            report_content.append("-" * 60)
            
            try:
                response = self.qa_chain.invoke({"query": question})
                report_content.append(f"Answer: {response['result']}")
                
                # Source documents removed for cleaner reports
                
            except Exception as e:
                report_content.append(f"Error: {e}")
            
            report_content.append("")
            report_content.append("=" * 80)
            report_content.append("")
        
        # Save report to file
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_content))
            
            console.print(f"[green]✓ Report saved successfully: {report_filename}[/green]")
            return report_filename
            
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")
            return ""
    
    def interactive_mode(self, gtp_analyzer: GTPAnalyzer):
        """Run interactive question-answering mode"""
        console.print("\n[bold cyan]Interactive GTP Analysis Mode[/bold cyan]")
        console.print("Type 'quit' or 'exit' to exit, 'help' for example questions")
        
        example_questions = [
            "What GTP tunnels are active in this capture?",
            "Are there any ICMP packets without responses?",
            "What is the source and destination of the encapsulated traffic?",
            "Are there any unusual GTP message types?",
            "What is the TEID distribution in this capture?",
            "How many ICMP echo requests are there?",
            "What extension headers are present in the GTP packets?",
            "Are there any patterns in the packet timing?"
        ]
        
        while True:
            try:
                question = input("\n[bold blue]Ask a question about the GTP packets: [/bold blue]")
                
                if question.lower() in ['quit', 'exit']:
                    console.print("\n[yellow]Exiting interactive mode...[/yellow]")
                    break
                elif question.lower() == 'help':
                    console.print("\n[bold yellow]Example Questions:[/bold yellow]")
                    for i, q in enumerate(example_questions, 1):
                        console.print(f"{i}. {q}")
                    continue
                elif not question.strip():
                    continue
                
                self.ask_question(question)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Exiting interactive mode...[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="GTP Packet Analysis using RAG")
    parser.add_argument("--pcap", required=True, help="Path to PCAP file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--report", action="store_true", help="Generate analysis report")
    parser.add_argument("--api-key", help="Google API key (or set GOOGLE_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Check if PCAP file exists
    if not os.path.exists(args.pcap):
        console.print(f"[red]Error: PCAP file {args.pcap} not found[/red]")
        return
    
    # Get API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your Google API Key: ")
    
    try:
        # Initialize GTP analyzer
        console.print("[bold blue]Initializing GTP Analyzer...[/bold blue]")
        gtp_analyzer = GTPAnalyzer()
        
        # Analyze PCAP file
        analysis_result = gtp_analyzer.analyze_pcap(args.pcap)
        
        if not analysis_result:
            console.print("[red]No analysis results generated[/red]")
            return
        
        # Print summary
        gtp_analyzer.print_summary()
        
        # Initialize RAG system
        console.print("\n[bold blue]Initializing RAG System...[/bold blue]")
        rag_system = GTPRAGSystem(api_key)
        
        # Create vector store
        rag_system.create_vectorstore(gtp_analyzer)
        
        # Setup QA chain
        rag_system.setup_qa_chain()
        
        # Run interactive mode or generate report
        if args.interactive:
            rag_system.interactive_mode(gtp_analyzer)
        elif args.report:
            # Generate comprehensive report
            console.print("\n[bold blue]Generating Analysis Report...[/bold blue]")
            
            report_file = rag_system.generate_report(args.pcap)
            if report_file:
                console.print(f"\n🎉 Report generation completed!")
                console.print(f"📄 Report saved to: {report_file}")
            else:
                console.print("\n❌ Report generation failed")
        else:
            # Default: ask a few key questions
            console.print("\n[bold blue]Default Analysis Questions:[/bold blue]")
            
            default_questions = [
                "What GTP tunnels are active in this capture?",
                "Are there any ICMP packets without responses?",
                "What is the TEID distribution in this capture?"
            ]
            
            for question in default_questions:
                rag_system.ask_question(question)
                console.print("\n" + "-"*50 + "\n")
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return


if __name__ == "__main__":
    main()
