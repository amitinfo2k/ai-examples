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
            features['total_packets'] += 1
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
            'icmp_echo_request_count': 0,
            'icmp_echo_reply_count': 0,
            'gtp_packets': 0,
            'gtp_teids': set(),
            'gtp_message_types': set(),
            'gtp_inner_protocols': set(),
            'gtp_icmp_packets': 0,
            'gtp_non_icmp_packets': 0,
            'pfcp_packets': 0,
            'pfcp_message_types': set(),
            'pfcp_cause_codes': set(),
            'pfcp_session_establishment_failed': False,
            'pfcp_heartbeat_only': False,
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
        
        # Get payload bytes once for both GTP and PFCP parsing
        payload_bytes = b''
        try:
            if Raw in pkt and hasattr(pkt[Raw], 'load'):
                payload_bytes = bytes(pkt[Raw].load)
            elif hasattr(udp, 'payload'):
                payload_bytes = bytes(udp.payload)
        except Exception:
            pass
        
        # Check for GTP-U (port 2152) or GTP-C (2123)
        if udp.dport in [2152, 2123] or udp.sport in [2152, 2123]:
            features['gtp_packets'] += 1
            self._parse_gtp_payload(payload_bytes, features)
        
        # PFCP (UDP 8805)
        if udp.dport == 8805 or udp.sport == 8805:
            features['pfcp_packets'] += 1
            self._parse_pfcp_payload(payload_bytes, features)
    
    def _parse_gtp_payload(self, payload_bytes: bytes, features: Dict[str, Any]) -> None:
        """Parse GTP payload to extract inner protocol information."""
        try:
            if len(payload_bytes) >= 8:
                # Check if this is GTP-U (protocol_type = 1)
                if payload_bytes[1] == 1:
                    # Skip GTP header (8 bytes) and try to detect inner IP protocol
                    inner_start = 8
                    if len(payload_bytes) > inner_start + 20:  # Minimum IP header
                        # IP version and protocol
                        ip_version = (payload_bytes[inner_start] >> 4) & 0xF
                        if ip_version == 4:
                            protocol = payload_bytes[inner_start + 9]
                            features['gtp_inner_protocols'].add(protocol)
                            if protocol == 1:  # ICMP
                                features['gtp_icmp_packets'] += 1
                            else:
                                features['gtp_non_icmp_packets'] += 1
        except Exception:
            pass
    
    def _parse_pfcp_payload(self, payload_bytes: bytes, features: Dict[str, Any]) -> None:
        """Parse PFCP payload to extract message types and cause codes."""
        try:
            if len(payload_bytes) >= 2:
                message_type = int(payload_bytes[1])
                features['pfcp_message_types'].add(message_type)
                
                # Parse Cause IE for session establishment failures
                if message_type == 51 and len(payload_bytes) >= 4:  # Session Establishment Response
                    self._parse_pfcp_cause_ie(payload_bytes, features)
        except Exception:
            pass
    
    def _parse_pfcp_cause_ie(self, payload_bytes: bytes, features: Dict[str, Any]) -> None:
        """Parse PFCP Cause IE to detect session establishment failures."""
        try:
            # Scan for IE Type bytes 0x00 0x13 (Cause IE)
            idx = 0
            while idx + 5 <= len(payload_bytes):
                if payload_bytes[idx] == 0x00 and payload_bytes[idx+1] == 0x13:
                    # length
                    length = (payload_bytes[idx+2] << 8) | payload_bytes[idx+3]
                    # cause value is typically 1 byte at start of IE value
                    if idx + 4 < len(payload_bytes):
                        cause_val = int(payload_bytes[idx+4])
                        features['pfcp_cause_codes'].add(cause_val)
                        # 73 -> Rule creation / modification Failure
                        if cause_val == 73:
                            features['pfcp_session_establishment_failed'] = True
                    break
                idx += 1
        except Exception:
            pass
    
    def _process_icmp_packet(self, pkt: Any, features: Dict[str, Any]) -> None:
        """Process ICMP packet."""
        features['protocol_counts']['ICMP'] += 1
        icmp_type = pkt[ICMP].type
        features['icmp_types'].add(icmp_type)
        if icmp_type == 8:
            features['icmp_echo_request_count'] += 1
        elif icmp_type == 0:
            features['icmp_echo_reply_count'] += 1
    
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
        features['gtp_inner_protocols'] = list(features.get('gtp_inner_protocols', []))
        features['pfcp_message_types'] = list(features['pfcp_message_types'])
        features['pfcp_cause_codes'] = list(features.get('pfcp_cause_codes', []))
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
        
        # Set PFCP heartbeat-only flag if only heartbeat messages present
        if features['pfcp_packets'] > 0 and len(features['pfcp_message_types']) == 2:
            if 1 in features['pfcp_message_types'] and 2 in features['pfcp_message_types']:
                features['pfcp_heartbeat_only'] = True
        
        return features
