import os
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP, SCTP, Raw, ICMP
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
from scapy.contrib.gtp import GTP_U_Header
from .pfcp_cause_codes import get_pfcp_cause_analyzer

class PCAPProcessor:
    """Process PCAP files to extract features and generate embeddings for RAG."""
    
    def __init__(self, config: dict, mapping_file: str = None):
        """Initialize the PCAP processor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
            mapping_file: Path to CSV file mapping PCAP filenames to labels
        """
        self.label_map = {}       
        self.config = config    
        self.logger = logging.getLogger(__name__)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pfcp_analyzer = get_pfcp_cause_analyzer()
        if mapping_file and os.path.exists(mapping_file):
            self.logger.info(f"Loading label mapping from {mapping_file}")
            self._load_label_mapping(mapping_file)
        else:
            self.logger.warning(f"Mapping file not found: {mapping_file}. Processing without labels.")
            mapping_file = None    
        
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
                'flow_stats': {},
                # New: ICMP stats
                'icmp_types': set(),
                'icmp_echo_request_count': 0,
                'icmp_echo_reply_count': 0,
                # New: GTP stats
                'gtp_packets': 0,
                'gtp_inner_protocols': set(),  # IP protocol numbers inside GTP-U
                'gtp_icmp_packets': 0,
                'gtp_non_icmp_packets': 0,
                # New: PFCP stats
                'pfcp_packets': 0,
                'pfcp_message_types': set(),   # numeric types (e.g., 1,2,50,51,...)
                'pfcp_cause_codes': set(),
                'pfcp_session_establishment_failed': False,
                'pfcp_session_modification_failed': False,
                'pfcp_session_deletion_failed': False,
                'pfcp_session_report_failed': False,
                'pfcp_heartbeat_only': False,
            }
            
            prev_time = None
            for pkt_index, pkt in enumerate(packets, start=1):
                # Protocol analysis
                if IP in pkt:
                    if TCP in pkt:
                        features['protocol_counts']['TCP'] += 1
                    elif UDP in pkt:
                        features['protocol_counts']['UDP'] += 1
                        udp = pkt[UDP]
                        # GTP-U detection by port (2152). GTP-C (2123) is control, skip inner parsing
                        if udp.dport == 2152 or udp.sport == 2152:
                            features['gtp_packets'] += 1
                            # Parse GTP-U inner payload using Scapy's GTP dissector
                            try:
                                g = GTP_U_Header(bytes(udp.payload))
                                inner = bytes(g.payload)
                                # Heuristic: find start of inner IP header if extensions remain
                                start = 0
                                if len(inner) >= 1:
                                    v_guess = (inner[0] >> 4) & 0x0F
                                    if v_guess not in (4, 6):
                                        # Scan first 64 bytes for a plausible IPv4/IPv6 header
                                        limit = min(len(inner), 96)
                                        found = -1
                                        for off in range(0, limit):
                                            b0 = inner[off]
                                            ver = (b0 >> 4) & 0x0F
                                            if ver == 4 and off + 20 <= len(inner):
                                                ihl = b0 & 0x0F
                                                if ihl >= 5 and off + ihl*4 <= len(inner):
                                                    found = off
                                                    break
                                            elif ver == 6 and off + 40 <= len(inner):
                                                found = off
                                                break
                                        if found >= 0:
                                            start = found
                                if len(inner) >= start + 1:
                                    v = (inner[start] >> 4) & 0x0F
                                    if v == 4 and len(inner) >= start + 20:
                                        ip_proto = inner[start + 9]
                                        features['gtp_inner_protocols'].add(int(ip_proto))
                                        if ip_proto == 1:  # ICMPv4
                                            features['gtp_icmp_packets'] += 1
                                            ihl_bytes = (inner[start] & 0x0F) * 4
                                            icmp_off = start + ihl_bytes
                                            if icmp_off < len(inner):
                                                icmp_type = int(inner[icmp_off])
                                                if icmp_type == 8:
                                                    features['icmp_echo_request_count'] += 1
                                                elif icmp_type == 0:
                                                    features['icmp_echo_reply_count'] += 1
                                        else:
                                            features['gtp_non_icmp_packets'] += 1
                                    elif v == 6 and len(inner) >= start + 40:
                                        next_header = inner[start + 6]
                                        features['gtp_inner_protocols'].add(int(next_header))
                                        if next_header == 58:  # ICMPv6
                                            features['gtp_icmp_packets'] += 1
                                        else:
                                            features['gtp_non_icmp_packets'] += 1
                            except Exception as ex:
                                self.logger.debug(f"[pkt {pkt_index}] GTP-U parse error (scapy): {ex}")
                        # PFCP (UDP 8805)
                        if udp.dport == 8805 or udp.sport == 8805:
                            features['pfcp_packets'] += 1
                            try:
                                payload_bytes = b''
                                if Raw in pkt and hasattr(pkt[Raw], 'load'):
                                    payload_bytes = bytes(pkt[Raw].load)
                                elif hasattr(udp, 'payload'):
                                    payload_bytes = bytes(udp.payload)
                                if len(payload_bytes) >= 2:
                                    pfcp_msg_type = int(payload_bytes[1])
                                    features['pfcp_message_types'].add(pfcp_msg_type)
                                    # Parse Cause IE for all PFCP response messages (51, 53, 55, 57, etc.)
                                    if pfcp_msg_type in [51, 53, 55, 57] and len(payload_bytes) >= 4:
                                        idx = 0
                                        while idx + 5 <= len(payload_bytes):
                                            if payload_bytes[idx] == 0x00 and payload_bytes[idx+1] == 0x13:
                                                if idx + 4 < len(payload_bytes):
                                                    cause_val = int(payload_bytes[idx+4])
                                                    features['pfcp_cause_codes'].add(cause_val)
                                                    # Enhanced failure detection using comprehensive cause code analysis
                                                    if self.pfcp_analyzer.is_rejection_cause(cause_val):
                                                        if pfcp_msg_type == 51:  # Session Establishment Response
                                                            features['pfcp_session_establishment_failed'] = True
                                                        elif pfcp_msg_type == 53:  # Session Modification Response
                                                            features['pfcp_session_modification_failed'] = True
                                                        elif pfcp_msg_type == 55:  # Session Deletion Response
                                                            features['pfcp_session_deletion_failed'] = True
                                                        elif pfcp_msg_type == 57:  # Session Report Response
                                                            features['pfcp_session_report_failed'] = True
                                                break
                                            idx += 1
                            except Exception:
                                pass
                    elif SCTP in pkt:
                        features['protocol_counts']['SCTP'] += 1
                        # NGAP runs over SCTP port 38412
                        try:
                            sctp = pkt[SCTP]
                            if getattr(sctp, 'sport', None) == 38412 or getattr(sctp, 'dport', None) == 38412:
                                features['ngap_messages'].append({
                                    'src_port': getattr(sctp, 'sport', None),
                                    'dst_port': getattr(sctp, 'dport', None),
                                    'length': len(sctp.payload) if hasattr(sctp, 'payload') else 0
                                })
                        except Exception:
                            pass
                    # Avoid double-counting: skip outer ICMP if this packet is GTP-U (we count inner ICMP instead)
                    if ICMP in pkt and not (UDP in pkt and (pkt[UDP].sport == 2152 or pkt[UDP].dport == 2152)):
                        icmp_type = int(pkt[ICMP].type)
                        features['icmp_types'].add(icmp_type)
                        if icmp_type == 8:
                            features['icmp_echo_request_count'] += 1
                        elif icmp_type == 0:
                            features['icmp_echo_reply_count'] += 1
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

            # PFCP heartbeat-only flag if only message types 1 & 2 present
            if features['pfcp_packets'] > 0 and len(features['pfcp_message_types']) == 2:
                if 1 in features['pfcp_message_types'] and 2 in features['pfcp_message_types']:
                    features['pfcp_heartbeat_only'] = True

            # Generate text description for RAG
            description = self._generate_description(features)
            features['description'] = description
            
            # Generate embedding for RAG
            features['embedding'] = self.embedding_model.encode(description).tolist()

            # Convert sets to lists for JSON serialization
            self._normalize_feature_types(features)

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
        
        # ICMP echo statistics
        if features.get('protocol_counts', {}).get('UDP', 0) or features.get('protocol_counts', {}).get('TCP', 0):
            if features.get('icmp_echo_request_count', 0) or features.get('icmp_echo_reply_count', 0):
                desc.append(
                    f"ICMP echo req/reply: {features.get('icmp_echo_request_count', 0)}/{features.get('icmp_echo_reply_count', 0)}."
                )

        # GTP summary
        if features.get('gtp_packets', 0) > 0:
            parts = [f"GTP packets: {features['gtp_packets']}"]
            inner = sorted(list(features.get('gtp_inner_protocols', [])))
            if inner:
                proto_names = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
                parts.append("inner=" + ", ".join([f"{p}:{proto_names.get(int(p), 'Unknown')}" for p in inner]))
            if features.get('gtp_icmp_packets', 0) or features.get('gtp_non_icmp_packets', 0):
                parts.append(f"ICMP:{features.get('gtp_icmp_packets', 0)} non-ICMP:{features.get('gtp_non_icmp_packets', 0)}")
            desc.append("; ".join(parts))

        # PFCP summary
        if features.get('pfcp_packets', 0) > 0:
            parts = [f"PFCP packets: {features['pfcp_packets']}"]
            types = sorted(list(features.get('pfcp_message_types', [])))
            if types:
                type_map = {1:'Heartbeat Req',2:'Heartbeat Resp',5:'Assoc Setup Req',6:'Assoc Setup Resp',50:'Sess Est Req',51:'Sess Est Resp',52:'Sess Mod Req',53:'Sess Mod Resp',54:'Sess Del Req',55:'Sess Del Resp'}
                parts.append("types=" + ", ".join([f"{t}:{type_map.get(int(t), 'Unknown')}" for t in types]))
            causes = sorted(list(features.get('pfcp_cause_codes', [])))
            if causes:
                # Use comprehensive cause code analysis
                cause_summary = self.pfcp_analyzer.get_cause_summary_text(causes)
                parts.append(f"causes: {cause_summary}")
            if features.get('pfcp_session_establishment_failed'):
                parts.append("session_establishment=FAILED")
            if features.get('pfcp_session_modification_failed'):
                parts.append("session_modification=FAILED")
            if features.get('pfcp_session_deletion_failed'):
                parts.append("session_deletion=FAILED")
            if features.get('pfcp_session_report_failed'):
                parts.append("session_report=FAILED")
            if features.get('pfcp_heartbeat_only'):
                parts.append("heartbeat_only")
            desc.append("; ".join(parts))

        # Error information
        if features['error_count'] > 0:
            desc.append(f"Found {features['error_count']} error(s).")
        
        return " ".join(desc)
    
    def _load_label_mapping(self, mapping_file: str) -> None:
        """Load label mapping from CSV file.
        
        Args:
            mapping_file: Path to CSV file with 'filename,label' format
        """
        try:
            self.logger.info(f"Loading label mapping from {mapping_file}")
            with open(mapping_file, 'r') as f:
                # Read all lines and strip whitespace
                lines = [line.strip() for line in f.readlines() if line.strip()]
                
                for line in lines:
                    # Split on comma and strip whitespace from each part
                    parts = [part.strip() for part in line.split(',') if part.strip()]
                    if len(parts) >= 2:  # Ensure we have both filename and label
                        filename = parts[0]
                        label = parts[1]
                        self.label_map[filename] = label
                        self.logger.debug(f"Mapped '{filename}' to label: {label}")
                    
            self.logger.info(f"Successfully loaded {len(self.label_map)} label mappings from {mapping_file}")
            
            # Debug: Log the loaded mappings
            if self.label_map:
                self.logger.debug("Loaded mappings:")
                for filename, label in self.label_map.items():
                    self.logger.debug(f"  {filename} -> {label}")
            else:
                self.logger.warning("No label mappings were loaded from the file")
                
        except Exception as e:
            self.logger.error(f"Failed to load label mapping from {mapping_file}: {str(e)}")
            self.label_map = {}
    
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
                # Add label if mapping exists
                pcap_name = pcap_file.name
                if pcap_name in self.label_map:
                    features['label'] = self.label_map[pcap_name]
                    self.logger.debug(f"Assigned label '{features['label']}' to {pcap_name}")
                else:
                    self.logger.warning(f"No label mapping found for {pcap_name}")
                    features['label'] = 'unknown'
                results.append(features)
            except Exception as e:
                self.logger.error(f"Failed to process {pcap_file}: {str(e)}")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Processed {len(results)} PCAP files. Results saved to {output_file}")
        return results

    def _normalize_feature_types(self, features: Dict) -> None:
        """Convert non-JSON-serializable types (e.g., sets) to lists in-place."""
        set_keys = [
            'icmp_types', 'gtp_inner_protocols', 'pfcp_message_types', 'pfcp_cause_codes'
        ]
        for key in set_keys:
            if isinstance(features.get(key), set):
                features[key] = list(features[key])
