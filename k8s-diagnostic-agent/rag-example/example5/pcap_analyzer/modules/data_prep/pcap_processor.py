import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP, SCTP
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm

class PCAPProcessor:
    """Process PCAP files to extract features and generate embeddings for RAG."""
    
    def __init__(self, config: dict):
        """Initialize the PCAP processor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def process_pcap(self, pcap_path: str) -> Dict:
        """Process a single PCAP file and extract features.
        
        Args:
            pcap_path: Path to the PCAP file
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        try:
            packets = rdpcap(pcap_path)
            features = {
                'pcap_name': os.path.basename(pcap_path),
                'total_packets': len(packets),
                'protocol_counts': {'TCP': 0, 'UDP': 0, 'SCTP': 0, 'Other': 0},
                'ngap_messages': [],
                'errors': [],
                'timings': [],
                'packet_sizes': [],
                'flow_stats': {}
            }
            
            prev_time = None
            for pkt in packets:
                # Protocol analysis
                if IP in pkt:
                    if TCP in pkt:
                        features['protocol_counts']['TCP'] += 1
                    elif UDP in pkt:
                        features['protocol_counts']['UDP'] += 1
                        # Check for NGAP (SCTP over UDP) or PFCP
                        if pkt[UDP].dport == 38412 or pkt[UDP].sport == 38412:  # NGAP
                            features['ngap_messages'].append({
                                'src_port': pkt[UDP].sport,
                                'dst_port': pkt[UDP].dport,
                                'length': len(pkt[UDP].payload)
                            })
                    elif SCTP in pkt:
                        features['protocol_counts']['SCTP'] += 1
                else:
                    features['protocol_counts']['Other'] += 1
                
                # Timing analysis
                if prev_time is not None:
                    features['timings'].append(float(pkt.time - prev_time))
                prev_time = pkt.time
                
                # Packet size analysis
                features['packet_sizes'].append(len(pkt))
                
                # TODO: Add more specific 5G protocol analysis
                
            # Calculate statistics
            features['avg_packet_size'] = np.mean(features['packet_sizes']) if features['packet_sizes'] else 0
            features['avg_timing'] = np.mean(features['timings']) if features['timings'] else 0
            features['error_count'] = len(features['errors'])
            features['ngap_message_count'] = len(features['ngap_messages'])
            
            # Generate text description for RAG
            description = self._generate_description(features)
            features['description'] = description
            
            # Generate embedding for RAG
            features['embedding'] = self.embedding_model.encode(description).tolist()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing {pcap_path}: {str(e)}")
            raise
    
    def _generate_description(self, features: Dict) -> str:
        """Generate a human-readable description of PCAP features.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            String description
        """
        desc = []
        desc.append(f"PCAP contains {features['total_packets']} packets.")
        
        # Protocol distribution
        protocols = ", ".join([f"{k}: {v}" for k, v in features['protocol_counts'].items()])
        desc.append(f"Protocol distribution: {protocols}")
        
        # NGAP specific
        if features['ngap_message_count'] > 0:
            desc.append(f"Contains {features['ngap_message_count']} potential NGAP messages.")
        
        # Error information
        if features['error_count'] > 0:
            desc.append(f"Found {features['error_count']} error(s).")
        
        return " ".join(desc)
    
    def process_directory(self, input_dir: str, output_file: str) -> None:
        """Process all PCAPs in a directory and save features to a file.
        
        Args:
            input_dir: Directory containing PCAP files
            output_file: Path to save processed features
        """
        pcap_files = list(Path(input_dir).glob('*.pcap'))
        if not pcap_files:
            self.logger.warning(f"No PCAP files found in {input_dir}")
            return
            
        results = []
        for pcap_file in tqdm(pcap_files, desc="Processing PCAPs"):
            try:
                features = self.process_pcap(str(pcap_file))
                results.append(features)
            except Exception as e:
                self.logger.error(f"Failed to process {pcap_file}: {str(e)}")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Processed {len(results)} PCAP files. Results saved to {output_file}")
        return results
