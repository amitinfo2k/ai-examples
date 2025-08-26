"""PCAP processing module for extracting features from network capture files."""
from pathlib import Path
from typing import Dict, Any, Set, List
import json
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP, SCTP, Raw, ICMP, Ether

class PCAPProcessor:
    """Processes PCAP files to extract features and metadata."""
    
    def process_pcap(self, pcap_path: str) -> Dict[str, Any]:
        """Process a PCAP file and extract relevant features."""
        try:
            packets = rdpcap(pcap_path)
        except Exception as e:
            raise ValueError(f"Failed to read PCAP file {pcap_path}: {str(e)}")
        
        features = self._init_features()
        prev_time = None
        
        for pkt in packets:
            if prev_time is not None:
                features['timings'].append(float(pkt.time - prev_time))
            prev_time = pkt.time
            
            features['packet_sizes'].append(len(pkt))
            
            if Ether in pkt and IP in pkt:
                self._process_ip_packet(pkt, features)
            else:
                features['protocol_counts']['Other'] += 1
        
        return self._finalize_features(features)
    
    def _init_features(self) -> Dict[str, Any]:
        """Initialize the features dictionary."""
        return {
            'total_packets': 0,
            'protocol_counts': {'TCP': 0, 'UDP': 0, 'SCTP': 0, 'ICMP': 0, 'Other': 0},
            'packet_sizes': [],
            'timings': [],
            'ip_ttl_values': [],
            'tcp_ports': set(),
            'udp_ports': set(),
            'icmp_types': set(),
            'gtp_packets': 0,
            'gtp_teids': set(),
            'gtp_message_types': set(),
            'pfcp_packets': 0,
            'pfcp_message_types': set(),
            'ngap_packets': 0,
            'ngap_procedure_codes': set(),
        }
    
    def _process_ip_packet(self, pkt: Any, features: Dict[str, Any]) -> None:
        """Process an IP packet and update features."""
        ip = pkt[IP]
        features['ip_ttl_values'].append(ip.ttl)
        
        if TCP in pkt:
            self._process_tcp_packet(pkt, features)
        elif UDP in pkt:
            self._process_udp_packet(pkt, features)
        elif ICMP in pkt:
            self._process_icmp_packet(pkt, features)
        elif SCTP in pkt:
            self._process_sctp_packet(pkt, features)
        else:
            features['protocol_counts']['Other'] += 1
    
    def _process_tcp_packet(self, pkt: Any, features: Dict[str, Any]) -> None:
        """Process TCP packet."""
        features['protocol_counts']['TCP'] += 1
        tcp = pkt[TCP]
        features['tcp_ports'].update([tcp.sport, tcp.dport])
    
    def _process_udp_packet(self, pkt: Any, features: Dict[str, Any]) -> None:
        """Process UDP packet."""
        features['protocol_counts']['UDP'] += 1
        udp = pkt[UDP]
        features['udp_ports'].update([udp.sport, udp.dport])
        
        # Check for GTP-U (port 2152) or GTP-C (2123)
        if udp.dport in [2152, 2123] or udp.sport in [2152, 2123]:
            features['gtp_packets'] += 1
    
    def _process_icmp_packet(self, pkt: Any, features: Dict[str, Any]) -> None:
        """Process ICMP packet."""
        features['protocol_counts']['ICMP'] += 1
        features['icmp_types'].add(pkt[ICMP].type)
    
    def _process_sctp_packet(self, pkt: Any, features: Dict[str, Any]) -> None:
        """Process SCTP packet."""
        features['protocol_counts']['SCTP'] += 1
        sctp = pkt[SCTP]
        if sctp.sport == 38412 or sctp.dport == 38412:
            features['ngap_packets'] += 1
    
    def _finalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize features by converting sets to lists and calculating statistics."""
        # Convert sets to lists for JSON serialization
        features['tcp_ports'] = list(features['tcp_ports'])
        features['udp_ports'] = list(features['udp_ports'])
        features['icmp_types'] = list(features['icmp_types'])
        features['gtp_teids'] = list(features['gtp_teids'])
        features['gtp_message_types'] = list(features['gtp_message_types'])
        features['pfcp_message_types'] = list(features['pfcp_message_types'])
        features['ngap_procedure_codes'] = list(features['ngap_procedure_codes'])
        
        # Calculate statistics
        features['avg_packet_size'] = float(np.mean(features['packet_sizes'])) if features['packet_sizes'] else 0.0
        features['std_packet_size'] = float(np.std(features['packet_sizes'])) if features['packet_sizes'] else 0.0
        features['avg_ttl'] = float(np.mean(features['ip_ttl_values'])) if features['ip_ttl_values'] else 0.0
        
        if features['timings']:
            features['avg_packet_rate'] = len(features['timings']) / sum(features['timings'])
            features['avg_packet_interval'] = float(np.mean(features['timings']))
        else:
            features['avg_packet_rate'] = 0.0
            features['avg_packet_interval'] = 0.0
        
        return features
