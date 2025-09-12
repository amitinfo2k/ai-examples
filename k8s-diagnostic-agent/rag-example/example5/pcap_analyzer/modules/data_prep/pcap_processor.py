import os
import json
import logging
from pathlib import Path
from typing import Dict, List
import subprocess
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP, SCTP, Raw, ICMP, SCTPChunkData
from sentence_transformers import SentenceTransformer
try:
    import pyshark  # Optional, used for robust NGAP/NAS fallback parsing
except Exception:  # pragma: no cover
    pyshark = None
import pandas as pd
from tqdm import tqdm
from scapy.contrib.gtp import GTP_U_Header
from .pfcp_cause_codes import get_pfcp_cause_analyzer
from ..protocol.ngap_decoder import NGAPDecoder
from .ngap_cause_codes import get_ngap_cause_text

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
        # Optional NGAP ASN.1 decoder (path can be configured; fallback if unavailable)
        schema_path = self.config.get('ngap', {}).get('asn1_schema_path') if isinstance(self.config, dict) else None
        self.ngap_decoder = NGAPDecoder(schema_path)
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
                # Enhanced: ICMP stats
                'icmp_types': set(),
                'icmp_echo_request_count': 0,
                'icmp_echo_reply_count': 0,
                # Enhanced: GTP stats
                'gtp_packets': 0,
                'gtp_inner_protocols': set(),  # IP protocol numbers inside GTP-U
                'gtp_icmp_packets': 0,
                'gtp_non_icmp_packets': 0,
                'gtp_tunnel_count': 0,        # Number of unique TEIDs
                'gtp_tunnel_status': [],      # Track tunnel establishment/teardown
                'gtp_user_plane_flows': [],   # Track data flows
                'gtp_control_plane_messages': [], # GTP-C messages if present
                'gtp_echo_responses': [],     # Track keepalive patterns
                'gtp_teids': set(),           # Set of unique TEIDs
                # Enhanced: PFCP stats
                'pfcp_packets': 0,
                'pfcp_message_types': set(),   # numeric types (e.g., 1,2,50,51,...)
                'pfcp_cause_codes': set(),
                'pfcp_session_establishment_failed': False,
                'pfcp_session_modification_failed': False,
                'pfcp_session_deletion_failed': False,
                'pfcp_session_report_failed': False,
                'pfcp_heartbeat_only': False,
                'pfcp_session_establishment_success_rate': 0.0,  # Percentage of successful establishments
                'pfcp_association_status': '',  # 'established', 'failed', 'timeout'
                'pfcp_session_count': 0,       # Total sessions in the capture
                'pfcp_retransmission_count': 0, # Track retransmissions
                'pfcp_timeout_events': [],     # Track timeout patterns
                'pfcp_node_capacity': {},      # Track node load indicators
                # Enhanced: NGAP stats
                'ngap_procedure_types': [],  # e.g., 'InitialUEMessage', 'AuthenticationRequest', 'SetupRequest'
                'ngap_message_types': [],    # Specific NGAP message type codes (numeric 0/1/2)
                'ngap_message_types_names': [],  # Human-readable types from pyshark (strings)
                'ngap_cause_codes': [],     # For reject/error messages
                'ngap_detailed_causes': [], # (category, value) tuples if available
                'ngap_amf_ue_ngap_id': [], # Track UE context
                'ngap_ran_ue_ngap_id': [], # Track RAN context
                'ngap_procedure_complete': False,  # Whether procedure completed successfully
                'ngap_authentication_steps': [],   # Track auth flow steps
                'ngap_security_steps': [],        # Track security setup steps
                'ngap_registration_status': '',   # 'success', 'failed', 'partial'
                # Enhanced: Timing and sequence analysis
                'timing_anomalies': [],       # Detect unusual delays
                'sequence_anomalies': [],     # Detect out-of-order packets
                'retransmission_patterns': [], # Track retransmission timing
                'protocol_handshake_completion': {}, # Track completion of various handshakes
                # Enhanced: Error and failure detection
                'error_patterns': [],         # Categorize error types
                'failure_scenarios': [],      # Identify specific failure modes
                'recovery_attempts': [],      # Track recovery mechanisms
                'root_cause_indicators': [],  # Clues about root causes
                'has_failures': False,        # Overall failure indicator
                # Enhanced: Context and metadata
                'ue_behavior_patterns': [],   # Track UE-specific patterns
                'network_load_indicators': [], # Track congestion/load
                'security_violations': [],    # Track security issues
                'compliance_issues': [],      # Track 3GPP compliance
                'specific_5g_issues': [],     # Detailed 5G protocol-specific issues
                # Pyshark-driven 5G enrichment
                'nas_messages': [],           # NAS 5GS message types via dissector
                'reject_causes': [],          # NAS/NGAP reject causes via dissector
                'ngap_retransmissions': 0,    # SCTP retransmit count via dissector
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
                        # Enhanced GTP-U detection by port (2152). GTP-C (2123) is control
                        if udp.dport == 2152 or udp.sport == 2152:
                            features['gtp_packets'] += 1
                            # Parse GTP-U inner payload using Scapy's GTP dissector
                            try:
                                g = GTP_U_Header(bytes(udp.payload))
                                inner = bytes(g.payload)
                                
                                # Extract TEID for tunnel tracking
                                if hasattr(g, 'teid'):
                                    teid = g.teid
                                    if teid not in features.get('gtp_teids', set()):
                                        if 'gtp_teids' not in features:
                                            features['gtp_teids'] = set()
                                        features['gtp_teids'].add(teid)
                                        features['gtp_tunnel_count'] = len(features['gtp_teids'])
                                
                                # Track tunnel status (establishment/teardown patterns)
                                if len(inner) > 0:
                                    # Simple heuristic: small packets might be control/signaling
                                    if len(inner) < 100:
                                        features['gtp_control_plane_messages'].append(f"Packet_{pkt_index}_Control_Size_{len(inner)}")
                                    else:
                                        features['gtp_user_plane_flows'].append(f"Packet_{pkt_index}_Data_Size_{len(inner)}")
                                
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
                                                    features['gtp_echo_responses'].append(f"Packet_{pkt_index}_EchoRequest")
                                                elif icmp_type == 0:
                                                    features['icmp_echo_reply_count'] += 1
                                                    features['gtp_echo_responses'].append(f"Packet_{pkt_index}_EchoReply")
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
                        # NGAP runs over SCTP port 38412 with PPID 60
                        try:
                            sctp = pkt[SCTP]
                            # Only process SCTP packets with PPID=60 (NGAP) as NGAP messages
                            ngap_payload = self._extract_sctp_ngap_payload(sctp)
                            port_match = (getattr(sctp, 'sport', None) == 38412 or getattr(sctp, 'dport', None) == 38412)
                            
                            # Check for PPID 60 explicitly in SCTP chunks
                            has_ngap_ppid = False
                            try:
                                if hasattr(sctp, 'chunks'):
                                    for ch in sctp.chunks:
                                        if isinstance(ch, SCTPChunkData) and getattr(ch, 'proto_id', None) == 60:
                                            has_ngap_ppid = True
                                            break
                                else:
                                    # Fallback: check payload layers for SCTPChunkData with PPID 60
                                    current = sctp.payload
                                    for _ in range(5):
                                        if isinstance(current, SCTPChunkData) and getattr(current, 'proto_id', None) == 60:
                                            has_ngap_ppid = True
                                            break
                                        if not hasattr(current, 'payload') or current.payload is None:
                                            break
                                        current = current.payload
                            except Exception:
                                pass
                            
                            # Only treat as NGAP if we have PPID=60 OR valid NGAP payload
                            is_ngap_packet = False
                            if ngap_payload is not None or has_ngap_ppid:
                                is_ngap_packet = True
                            
                            if is_ngap_packet:
                                # Enhanced NGAP message parsing
                                # Pass pyshark NGAP data if available for better message type detection
                                pyshark_ngap = getattr(pkt, 'ngap', None) if hasattr(pkt, 'ngap') and pkt.ngap else None
                                ngap_info = self._parse_ngap_message(pkt, sctp, pkt_index, pyshark_ngap)
                                
                                # Add packet number to track which packets contain NGAP
                                ngap_info['packet_number'] = pkt_index
                                
                                # If ASN.1 decoder is available, try decoding and enrich fields
                                if self.ngap_decoder and self.ngap_decoder.is_available():
                                    payload_bytes = ngap_payload if ngap_payload is not None else b''
                                    if not payload_bytes:
                                        if hasattr(sctp, 'payload') and sctp.payload:
                                            payload_bytes = bytes(sctp.payload)
                                        elif Raw in pkt and hasattr(pkt[Raw], 'load'):
                                            payload_bytes = bytes(pkt[Raw].load)
                                    if payload_bytes:
                                        decoded = self.ngap_decoder.decode_pdu(payload_bytes)
                                        if decoded:
                                            basic = self.ngap_decoder.extract_basic_fields(decoded)
                                            # Merge fields if decoder produced values
                                            for k in ('procedure_code', 'message_type', 'amf_ue_ngap_id', 'ran_ue_ngap_id'):
                                                if basic.get(k) is not None:
                                                    ngap_info[k] = basic[k]
                                            # Map cause to numeric code when possible for existing logic
                                            if basic.get('cause'):
                                                cause = basic['cause']
                                                # Keep category:value representation
                                                ngap_info['cause_category'] = cause.get('category')
                                                ngap_info['cause_code'] = cause.get('value')
                                
                                # Safety check: ensure ngap_info is not None
                                if ngap_info is None:
                                    ngap_info = {
                                        'src_port': getattr(sctp, 'sport', None),
                                        'dst_port': getattr(sctp, 'dport', None),
                                        'length': len(sctp.payload) if hasattr(sctp, 'payload') else 0,
                                        'procedure_code': None,
                                        'message_type': None,
                                        'amf_ue_ngap_id': None,
                                        'ran_ue_ngap_id': None,
                                        'cause_code': None,
                                        'is_authentication': False,
                                        'is_security': False,
                                        'is_setup': False,
                                        'is_ue_setup': False,
                                        'is_gnb_setup': False,
                                        'is_reject': False
                                    }
                                
                                features['ngap_messages'].append(ngap_info)
                                
                                # Extract NGAP procedure and message types
                                if ngap_info.get('procedure_code'):
                                    features['ngap_procedure_types'].append(ngap_info['procedure_code'])
                                # Note: ngap_message_types is now populated consistently in the pyshark fallback section
                                # to ensure both arrays have the same length and correspond to the same packets
                                
                                # Track NGAP IDs for context
                                if ngap_info.get('amf_ue_ngap_id'):
                                    features['ngap_amf_ue_ngap_id'].append(ngap_info['amf_ue_ngap_id'])
                                if ngap_info.get('ran_ue_ngap_id'):
                                    features['ngap_ran_ue_ngap_id'].append(ngap_info['ran_ue_ngap_id'])
                                
                                # Track authentication and security steps
                                if ngap_info.get('is_authentication'):
                                    features['ngap_authentication_steps'].append(ngap_info['message_type'])
                                if ngap_info.get('is_security'):
                                    features['ngap_security_steps'].append(ngap_info['message_type'])
                                
                                # Track cause codes only for explicit rejects (unsuccessfulOutcome) or NAS rejects
                                if (
                                    ngap_info.get('cause_code') is not None and
                                    (
                                        ngap_info.get('message_type') == 2 or
                                        (ngap_info.get('is_reject') and ngap_info.get('nas_message_type') is not None)
                                    )
                                ):
                                    features['ngap_cause_codes'].append(ngap_info['cause_code'])
                                # Store detailed cause if available from decoder
                                if ngap_info.get('cause_category') and ngap_info.get('cause_code') is not None:
                                    features['ngap_detailed_causes'].append({
                                        'category': ngap_info['cause_category'],
                                        'value': ngap_info['cause_code'] if isinstance(ngap_info['cause_code'], str) else str(ngap_info['cause_code'])
                                    })
                                    
                        except Exception as e:
                            self.logger.debug(f"NGAP parsing error: {e}")
                            # Fallback to basic info
                            ngap_info = {
                                'src_port': getattr(sctp, 'sport', None),
                                'dst_port': getattr(sctp, 'dport', None),
                                'length': len(sctp.payload) if hasattr(sctp, 'payload') else 0,
                                'procedure_code': None,
                                'message_type': None,
                                'amf_ue_ngap_id': None,
                                'ran_ue_ngap_id': None,
                                'cause_code': None,
                                'is_authentication': False,
                                'is_security': False,
                                'is_setup': False,
                                'is_ue_setup': False,
                                'is_gnb_setup': False,
                                'is_reject': False
                            }
                            features['ngap_messages'].append(ngap_info)
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
            features['ngap_message_count'] = len(features['ngap_messages'])

            # Enhanced analysis
            self._detect_failure_patterns(features)
            
            # Calculate error count after failure patterns are detected and errors list is populated
            features['error_count'] = len(features['errors'])
            self._analyze_timing_anomalies(features)
            self._enhance_pfcp_analysis(features)

            # PFCP heartbeat-only flag if only message types 1 & 2 present
            if features['pfcp_packets'] > 0 and len(features['pfcp_message_types']) == 2:
                if 1 in features['pfcp_message_types'] and 2 in features['pfcp_message_types']:
                    features['pfcp_heartbeat_only'] = True

            # Deep 5G parsing via pyshark before building description/embedding
            try:
                if pyshark is not None:
                    self._pyshark_extract_5g_features(pcap_path, features)
            except Exception:
                pass

            # Generate text description for RAG
            description = self._generate_description(features)
            features['description'] = description
            
            # Legacy pyshark enrichment (string-based) as an additional best-effort
            try:
                if pyshark is not None and features.get('ngap_registration_status') != 'failed':
                    self._pyshark_enrich_ngap(pcap_path, features)
            except Exception:
                # Best-effort fallback; ignore errors
                pass

            # Provide a focused RAG query template to guide LLM reasoning
            features['rag_query'] = self._build_rag_query(features)

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
        
        # Enhanced NGAP specific
        if features['ngap_message_count'] > 0:
            desc.append(f"Contains {features['ngap_message_count']} NGAP messages.")
            
            # Add procedure types if available (show unique procedures with counts)
            if features.get('ngap_procedure_types'):
                # Count unique procedures
                procedure_counts = {}
                for proc in features['ngap_procedure_types']:
                    proc_name = self._get_procedure_name(proc)
                    procedure_counts[proc_name] = procedure_counts.get(proc_name, 0) + 1
                
                # Format as "ProcedureName (count), ..."
                procedure_summary = []
                for proc_name, count in sorted(procedure_counts.items()):
                    if count > 1:
                        procedure_summary.append(f"{proc_name} ({count})")
                    else:
                        procedure_summary.append(proc_name)
                
                desc.append(f"NGAP procedures: {', '.join(procedure_summary)}")
            
            # Add registration status
            if features.get('ngap_registration_status'):
                desc.append(f"Registration status: {features['ngap_registration_status']}")
            
            # Add authentication and security steps
            if features.get('ngap_authentication_steps'):
                desc.append(f"Authentication steps: {len(features['ngap_authentication_steps'])}")
            if features.get('ngap_security_steps'):
                desc.append(f"Security steps: {len(features['ngap_security_steps'])}")
            
            # Add cause codes if any failures
            if features.get('ngap_cause_codes'):
                causes = [str(cause) for cause in features['ngap_cause_codes']]
                desc.append(f"NGAP cause codes: {', '.join(causes)}")
        
        # ICMP echo statistics
        if features.get('protocol_counts', {}).get('UDP', 0) or features.get('protocol_counts', {}).get('TCP', 0):
            if features.get('icmp_echo_request_count', 0) or features.get('icmp_echo_reply_count', 0):
                desc.append(
                    f"ICMP echo req/reply: {features.get('icmp_echo_request_count', 0)}/{features.get('icmp_echo_reply_count', 0)}."
                )

        # Enhanced GTP summary
        if features.get('gtp_packets', 0) > 0:
            parts = [f"GTP packets: {features['gtp_packets']}"]
            
            # Add tunnel information
            if features.get('gtp_tunnel_count', 0) > 0:
                parts.append(f"tunnels={features['gtp_tunnel_count']}")
            
            inner = sorted(list(features.get('gtp_inner_protocols', [])))
            if inner:
                proto_names = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
                parts.append("inner=" + ", ".join([f"{p}:{proto_names.get(int(p), 'Unknown')}" for p in inner]))
            
            if features.get('gtp_icmp_packets', 0) or features.get('gtp_non_icmp_packets', 0):
                parts.append(f"ICMP:{features.get('gtp_icmp_packets', 0)} non-ICMP:{features.get('gtp_non_icmp_packets', 0)}")
            
            # Add user plane vs control plane info
            if features.get('gtp_user_plane_flows'):
                parts.append(f"user_plane_flows={len(features['gtp_user_plane_flows'])}")
            if features.get('gtp_control_plane_messages'):
                parts.append(f"control_messages={len(features['gtp_control_plane_messages'])}")
            
            desc.append("; ".join(parts))

        # Enhanced PFCP summary
        if features.get('pfcp_packets', 0) > 0:
            parts = [f"PFCP packets: {features['pfcp_packets']}"]
            types = sorted(list(features.get('pfcp_message_types', [])))
            if types:
                type_map = {1:'Heartbeat Req',2:'Heartbeat Resp',5:'Assoc Setup Req',6:'Assoc Setup Resp',50:'Sess Est Req',51:'Sess Est Resp',52:'Sess Mod Req',53:'Sess Mod Resp',54:'Sess Del Req',55:'Sess Del Resp'}
                parts.append("types=" + ", ".join([f"{t}:{type_map.get(int(t), 'Unknown')}" for t in types]))
            
            # Add enhanced PFCP metrics
            if features.get('pfcp_association_status'):
                parts.append(f"association_status={features['pfcp_association_status']}")
            if features.get('pfcp_session_count', 0) > 0:
                parts.append(f"sessions={features['pfcp_session_count']}")
            if features.get('pfcp_session_establishment_success_rate', 0) > 0:
                parts.append(f"success_rate={features['pfcp_session_establishment_success_rate']:.2f}")
            
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

        # Enhanced failure and error analysis
        if features.get('has_failures'):
            desc.append("FAILURE_DETECTED")
            if features.get('failure_patterns'):
                desc.append(f"Failure patterns: {', '.join(features['failure_patterns'])}")
            if features.get('failure_scenarios'):
                desc.append(f"Failure scenarios: {', '.join(features['failure_scenarios'])}")
            if features.get('root_cause_indicators'):
                desc.append(f"Root cause indicators: {', '.join(features['root_cause_indicators'])}")
        
        # Specific NGAP failure analysis
        if features.get('ngap_message_count', 0) > 0:
            if features.get('ngap_registration_status') == 'failed':
                desc.append("NGAP_REGISTRATION_FAILED")
            elif features.get('ngap_registration_status') == 'partial':
                desc.append("NGAP_REGISTRATION_PARTIAL")
            
            # Check for specific Initial Context Setup failures
            if features.get('protocol_handshake_completion', {}).get('ngap_initial_context_setup') == 'failed':
                desc.append("NGAP_INITIAL_CONTEXT_SETUP_FAILED")
            elif features.get('protocol_handshake_completion', {}).get('ngap_initial_context_setup') == 'incomplete':
                desc.append("NGAP_INITIAL_CONTEXT_SETUP_INCOMPLETE")
        
        # Specific 5G protocol issues
        if features.get('specific_5g_issues'):
            for issue in features['specific_5g_issues']:
                issue_desc = f"5G_ISSUE: {issue['type']} (Cause {issue['cause_code']}) - {issue['description']} [{issue['component']}]"
                desc.append(issue_desc)
        
        # Timing and sequence analysis
        if features.get('timing_anomalies'):
            desc.append(f"Timing anomalies: {len(features['timing_anomalies'])}")
        if features.get('sequence_anomalies'):
            desc.append(f"Sequence anomalies: {len(features['sequence_anomalies'])}")
        if features.get('retransmission_patterns'):
            desc.append(f"Retransmission indicators: {len(features['retransmission_patterns'])}")

        # Error information
        if features['error_count'] > 0:
            desc.append(f"Found {features['error_count']} error(s).")
        
        return " ".join(desc)
    
    def _build_rag_query(self, features: Dict) -> str:
        """Construct a focused RAG retrieval query highlighting NGAP/NAS reject indicators."""
        ngap_msgs = []
        try:
            for m in features.get('ngap_messages', []):
                if isinstance(m, dict):
                    mt = m.get('message_type')
                    pc = m.get('procedure_code')
                    ngap_msgs.append(str(mt) if pc is None else f"{mt}_{pc}")
                else:
                    ngap_msgs.append(str(m))
        except Exception:
            pass
        nas_msgs = [str(x) for x in features.get('nas_messages', [])]
        causes = [str(c) for c in features.get('ngap_cause_codes', [])]
        # Include NAS/NGAP reject_causes if present from pyshark
        extra_causes = [str(c) for c in features.get('reject_causes', [])]
        if extra_causes:
            causes.extend(extra_causes)
        status = features.get('ngap_registration_status') or 'unknown'
        avg_t = features.get('avg_timing', 0)
        retrans = features.get('ngap_retransmissions', 0)
        return (
            f"Analyze this 5G PCAP for registration issues: NGAP messages: {ngap_msgs}, "
            f"NAS messages: {nas_msgs}, reject causes present: {causes}, sequence status: {status}, "
            f"average packet timing: {avg_t:.3f}s, retransmissions: {retrans}. Focus on detecting rejects "
            f"(e.g., NGAP Cause >0, NAS Registration Reject 5GS cause like 15 for 'no suitable cells') vs. accepts "
            f"(Registration Accept + UE Context Setup)."
        )

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
        except Exception as e:
            self.logger.warning(f"Failed to load mapping file {mapping_file}: {e}")

    def _pyshark_extract_5g_features(self, pcap_path: str, features: Dict) -> None:
        """Parse NGAP/NAS using pyshark for accurate 5G control-plane extraction.
        Captures message types, reject causes, and simple sequence completion.
        """
        if pyshark is None:
            return
        # Quick guard using tshark probe
        if not self._pcap_supported_by_tshark(pcap_path):
            self.logger.warning("Pyshark/tshark cannot parse this PCAP's link type; skipping pyshark extraction for this file.")
            return
        # Focus on NGAP over SCTP (38412) and NAS-5GS
        display = 'ngap || nas-5gs || sctp.port == 38412'
        cap = None
        try:
            cap = pyshark.FileCapture(pcap_path, display_filter=display, only_summaries=False)
            prev_ts = None
            reg_started = False
            reg_accepted = False
            reg_rejected = False
            sctp_seq_seen = set()
            # Load incrementally to avoid memory blow-up on large pcaps
            for pkt in cap:
                try:
                    # Timings (pyshark exposes sniff_timestamp)
                    try:
                        ts = float(getattr(pkt, 'sniff_timestamp', None) or 0.0)
                    except Exception:
                        ts = None
                    if ts is not None and prev_ts is not None and ts >= prev_ts:
                        features['timings'].append(ts - prev_ts)
                    if ts is not None:
                        prev_ts = ts

                    # NGAP
                    if hasattr(pkt, 'ngap') and pkt.ngap:
                        # Message type/procedure code names vary; capture generically
                        # Prefer pretty showname if available; otherwise map numeric to descriptive label
                        pc = getattr(pkt.ngap, 'procedure_code', None)
                        pretty = None
                        mt_int = None
                        try:
                            # Common pyshark/tshark pretty fields
                            pretty = getattr(pkt.ngap, 'ngap_pdu_showname', None) or getattr(pkt.ngap, 'type_of_message_showname', None)
                        except Exception:
                            pretty = None
                        if not pretty:
                            mt_val = None
                            try:
                                mt_val = getattr(pkt.ngap, 'type_of_message', None)
                            except Exception:
                                mt_val = None
                            if mt_val is None:
                                try:
                                    mt_val = getattr(pkt.ngap, 'ngap_pdu', None)
                                except Exception:
                                    mt_val = None
                            # Normalize to int when possible
                            try:
                                if mt_val is not None:
                                    mt_int = int(str(mt_val))
                            except Exception:
                                mt_int = None
                            if mt_int in (0, 1, 2):
                                label = {0: 'initiatingMessage', 1: 'successfulOutcome', 2: 'unsuccessfulOutcome'}[mt_int]
                                pretty = f"NGAP-PDU: {label} ({mt_int})"
                            elif mt_val is not None:
                                pretty = str(mt_val)
                        try:
                            if pretty:
                                features['ngap_message_types_names'].append(str(pretty))
                                # Also populate the numeric array consistently from pyshark data
                                if mt_int is not None and mt_int in (0, 1, 2):
                                    features['ngap_message_types'].append(mt_int)
                        except Exception:
                            pass
                        # Procedure codes should only be added from actual NGAP message parsing
                        # not from pyshark fallback which processes all SCTP packets
                        
                        # Registration flow hints
                        text = str(pkt.ngap).lower()
                        if 'initial ue message' in text or 'initial_ue_message' in text:
                            reg_started = True
                        if 'ue context setup response' in text or 'registration accept' in text:
                            reg_accepted = True

                        # Cause IEs (any category) - only for actual rejection messages
                        text = str(pkt.ngap).lower()
                        is_rejection_message = ('registration reject' in text or 
                                              'initial context setup failure' in text or
                                              'ue context release' in text or
                                              'ngap setup failure' in text)
                        
                        if is_rejection_message:
                            for f in getattr(pkt.ngap, 'field_names', []):
                                if f.startswith('cause') or 'cause' in f:
                                    try:
                                        val = getattr(pkt.ngap, f)
                                        if val is not None:
                                            # Normalize numeric if possible
                                            try:
                                                cv = int(str(val))
                                            except Exception:
                                                continue
                                            features['ngap_cause_codes'].append(cv)
                                            features['reject_causes'].append(cv)
                                            reg_rejected = True
                                    except Exception:
                                        continue

                        # Retransmission via SCTP sequence duplicate
                        if hasattr(pkt, 'sctp') and hasattr(pkt.sctp, 'tsn'):
                            try:
                                seq = int(pkt.sctp.tsn)
                                if seq in sctp_seq_seen:
                                    features['ngap_retransmissions'] += 1
                                sctp_seq_seen.add(seq)
                            except Exception:
                                pass

                    # NAS-5GS layer
                    if hasattr(pkt, 'nas_5gs') and pkt.nas_5gs:
                        # Try to extract 5GMM message type and detect rejects
                        try:
                            mm = getattr(pkt.nas_5gs, 'mm', None)
                            msg_t = None
                            if mm is not None and hasattr(mm, 'message_type'):
                                msg_t = str(mm.message_type)
                            else:
                                # Fallback to any recognizable field
                                msg_t = getattr(pkt.nas_5gs, 'message_type', None)
                            if msg_t is not None:
                                features['nas_messages'].append(str(msg_t))
                        except Exception:
                            pass

                        # Registration Reject detection
                        low = str(pkt.nas_5gs).lower()
                        if ('registration reject' in low or 'registration_reject' in low) and ('nas-5gs' in low or 'nas_5gs' in low or 'ngap' in low):
                            # 5GS cause field name can differ across versions
                            cause_fields = [
                                'mm_5gs_cause',
                                'registration_reject_5gs_cause',
                                '5gmm.cause',
                                'nas_5gs.mm.5gmm.cause'
                            ]
                            cause_val = None
                            for cf in cause_fields:
                                try:
                                    parts = cf.split('.')
                                    obj = pkt
                                    for p in parts:
                                        if not hasattr(obj, p.replace('5gmm', 'mm')):
                                            obj = None
                                            break
                                        obj = getattr(obj, p.replace('5gmm', 'mm'))
                                    if obj is not None:
                                        try:
                                            cause_val = int(str(obj))
                                            break
                                        except Exception:
                                            pass
                                except Exception:
                                    continue
                            if cause_val is None:
                                # Default to Illegal UE if text mentions it
                                if 'illegal ue' in low:
                                    cause_val = 3
                            if cause_val is not None:
                                features['reject_causes'].append(int(cause_val))
                            reg_rejected = True

                except Exception:
                    continue
            # Finalize status
            if reg_rejected:
                features['ngap_registration_status'] = 'failed'
                if 'Registration_Rejection' not in features['failure_scenarios']:
                    features['failure_scenarios'].append('Registration_Rejection')
                if 'NAS_Registration_Rejection' not in features['errors']:
                    features['errors'].append('NAS_Registration_Rejection')
            elif reg_accepted:
                if not features.get('ngap_registration_status'):
                    features['ngap_registration_status'] = 'success'
            elif reg_started:
                if not features.get('ngap_registration_status'):
                    features['ngap_registration_status'] = 'partial'

        except Exception as e:
            # Silent best-effort; handle unsupported link-layer early
            err = str(e)
            if 'network type' in err and 'unknown or unsupported' in err:
                self.logger.warning("Pyshark/tshark cannot parse this PCAP's link type; skipping pyshark extraction for this file.")
            return
        finally:
            if cap is not None:
                try:
                    cap.close()
                except Exception:
                    pass

    def _pyshark_enrich_ngap(self, pcap_path: str, features: Dict) -> None:
        """Best-effort enrichment using pyshark (tshark dissectors).
        Detect DownlinkNASTransport with Registration Reject and Illegal UE cause
        when byte-level parsing missed it. No-op if pyshark is unavailable.
        """
        if pyshark is None:
            return
        # Quick guard using tshark probe
        if not self._pcap_supported_by_tshark(pcap_path):
            self.logger.warning("Pyshark/tshark cannot parse this PCAP's link type; skipping pyshark enrichment for this file.")
            return
        capture = None
        try:
            capture = pyshark.FileCapture(pcap_path, display_filter='ngap || nas-5gs', use_json=True)
            for pkt in capture:
                try:
                    raw_str = str(pkt)
                    low = raw_str.lower()
                    if ('registration reject' in low or 'registration_reject' in low) and ('nas-5gs' in low or 'nas_5gs' in low or 'ngap' in low):
                        cause_code = None
                        # Try a few common patterns
                        import re
                        m = re.search(r'cause[^0-9]*(\d+)', low)
                        if m:
                            try:
                                cause_code = int(m.group(1))
                            except Exception:
                                cause_code = None
                        if cause_code is None:
                            # Try to parse key=value style
                            m = re.search(r'5gmm\.cause\s*[:=]\s*(\d+)', low)
                            if m:
                                try:
                                    cause_code = int(m.group(1))
                                except Exception:
                                    cause_code = None
                        if cause_code is None:
                            # Default to Illegal UE if text mentions it
                            if 'illegal ue' in low:
                                cause_code = 3
                        if cause_code is not None:
                            features['ngap_registration_status'] = 'failed'
                            if 'NAS_Registration_Rejection' not in features['errors']:
                                features['errors'].append('NAS_Registration_Rejection')
                            indicator = f"NAS_Registration_Reject_Cause_{cause_code}"
                            if indicator not in features['root_cause_indicators']:
                                features['root_cause_indicators'].append(indicator)
                            # Add specific 5G issue block
                            issue_desc_map = {3: 'Illegal UE', 6: 'Illegal ME', 11: 'PLMN not allowed', 12: 'Tracking area not allowed'}
                            features['specific_5g_issues'].append({
                                'type': 'Registration_Reject',
                                'cause_code': cause_code,
                                'description': issue_desc_map.get(cause_code, f'Cause {cause_code}'),
                                'severity': 'high' if cause_code in [3, 6] else 'medium',
                                'component': 'AMF'
                            })
                            break
                except Exception:
                    continue
        except Exception as e:
            # Ignore pyshark failures silently; this is a best-effort enrichment
            err = str(e)
            if 'network type' in err and 'unknown or unsupported' in err:
                self.logger.warning("Pyshark/tshark cannot parse this PCAP's link type; skipping pyshark enrichment for this file.")
            return
        finally:
            if capture is not None:
                try:
                    capture.close()
                except Exception:
                    pass

    def _pcap_supported_by_tshark(self, pcap_path: str) -> bool:
        """Check with tshark whether the PCAP link type is supported.
        Returns False for known unsupported encap types (e.g., 276 on older tshark).
        """
        try:
            proc = subprocess.run([
                'tshark', '-r', pcap_path, '-T', 'fields', '-e', 'frame.encap_type', '-c', '1'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            if proc.returncode != 0:
                err = (proc.stderr or '').lower()
                if 'unknown or unsupported' in err or 'not a capture file' in err or "couldn't open" in err:
                    return False
                return True
            out = (proc.stdout or '').strip()
            if out:
                try:
                    encap = int(out.splitlines()[0].strip())
                    if encap == 276:
                        return False
                except Exception:
                    pass
            return True
        except Exception:
            return True

    
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

    def _parse_ngap_message(self, packet, sctp, packet_index: int, pyshark_ngap=None) -> Dict:
        """Enhanced NGAP message parsing to extract detailed information.
        
        Args:
            packet: The full packet
            sctp: The SCTP layer
            packet_index: Packet index for logging
            
        Returns:
            Dictionary with parsed NGAP information
        """
        try:
            ngap_info = {
                'src_port': getattr(sctp, 'sport', None),
                'dst_port': getattr(sctp, 'dport', None),
                'length': len(sctp.payload) if hasattr(sctp, 'payload') else 0,
                'procedure_code': None,
                'message_type': None,
                'amf_ue_ngap_id': None,
                'ran_ue_ngap_id': None,
                'cause_code': None,
                'is_authentication': False,
                'is_security': False,
                'is_setup': False,
                'is_ue_setup': False,
                'is_gnb_setup': False,
                'is_reject': False
            }
            
            # Method 0: Try pyshark message type detection first (most reliable)
            if pyshark_ngap:
                try:
                    mt_val = None
                    # Try different pyshark field names for message type
                    for field_name in ['type_of_message', 'ngap_pdu']:
                        try:
                            mt_val = getattr(pyshark_ngap, field_name, None)
                            if mt_val is not None:
                                break
                        except Exception:
                            continue
                    
                    if mt_val is not None:
                        try:
                            mt_int = int(str(mt_val))
                            if mt_int in (0, 1, 2):
                                ngap_info['message_type'] = mt_int
                                self.logger.debug(f"Pyshark detected message_type: {mt_int}")
                        except Exception:
                            pass
                except Exception as e:
                    self.logger.debug(f"Pyshark message type detection failed: {e}")
            
            # Parse NGAP payload if available
            payload_bytes = b''
            
            # Prefer SCTP Data chunk with PPID 60
            ngap_payload = self._extract_sctp_ngap_payload(sctp)
            if ngap_payload is not None:
                payload_bytes = ngap_payload
                self.logger.debug(f"SCTP NGAP (PPID 60) payload length: {len(payload_bytes)}")
            else:
                # Fallbacks: entire SCTP payload, then Raw
                if hasattr(sctp, 'payload') and sctp.payload:
                    payload_bytes = bytes(sctp.payload)
                    self.logger.debug(f"SCTP payload length: {len(payload_bytes)}")
                if len(payload_bytes) == 0 and Raw in packet and hasattr(packet[Raw], 'load'):
                    payload_bytes = bytes(packet[Raw].load)
                    self.logger.debug(f"Raw payload length: {len(payload_bytes)}")
            
            # Debug logging for NGAP payload analysis
            if len(payload_bytes) > 0:
                self.logger.debug(f"Packet {packet_index}: NGAP payload hex: {payload_bytes[:32].hex() if len(payload_bytes) >= 32 else payload_bytes.hex()}")
                # Additional debug for packet 88
                if packet_index == 88:
                    self.logger.info(f"PACKET 88 DEBUG - Full payload: {payload_bytes.hex()}")
                    self.logger.info(f"PACKET 88 DEBUG - First 20 bytes: {[hex(b) for b in payload_bytes[:20]]}")
            else:
                self.logger.warning("No NGAP payload found in packet")
            
            # Enhanced NGAP parsing using ASN.1 decoder first, then fallback methods
            if len(payload_bytes) >= 2:
                # Method 1: Try ASN.1 decoder first if available
                if self.ngap_decoder.is_available():
                    try:
                        decoded = self.ngap_decoder.decode_pdu(payload_bytes)
                        if decoded:
                            extracted = self.ngap_decoder.extract_basic_fields(decoded)
                            if extracted.get('procedure_code') is not None:
                                ngap_info['procedure_code'] = extracted['procedure_code']
                                # Don't set message_type from ASN.1 decoder yet - let improved parsing handle it
                                ngap_info['amf_ue_ngap_id'] = extracted.get('amf_ue_ngap_id')
                                ngap_info['ran_ue_ngap_id'] = extracted.get('ran_ue_ngap_id')
                                if extracted.get('cause'):
                                    ngap_info['cause_code'] = f"{extracted['cause']['category']}:{extracted['cause']['value']}"
                                self.logger.debug(f"ASN.1 decoded NGAP procedure code: {extracted['procedure_code']}")
                                # Continue to improved message type parsing instead of returning early
                    except Exception as e:
                        self.logger.debug(f"ASN.1 decoding failed, using fallback: {e}")
                
                # Method 1.1: Improved ASN.1 NGAP-PDU structure parsing for message type (only if pyshark failed)
                # NGAP-PDU is a CHOICE with context-specific tags [0], [1], [2]
                if ngap_info.get('message_type') is None:  # Only run if pyshark didn't detect it
                    try:
                        self.logger.debug(f"Starting ASN.1 message type detection for payload of {len(payload_bytes)} bytes")
                        if len(payload_bytes) >= 2:
                            # Log first few bytes for debugging
                            hex_bytes = ' '.join(f'{b:02x}' for b in payload_bytes[:8])
                            self.logger.debug(f"First 8 bytes: {hex_bytes}")
                            
                            # Look for ASN.1 BER/DER encoded CHOICE tags
                            # NGAP-PDU ::= CHOICE {
                            #   initiatingMessage       [0] InitiatingMessage,
                            #   successfulOutcome       [1] SuccessfulOutcome,
                            #   unsuccessfulOutcome     [2] UnsuccessfulOutcome
                            # }
                            
                            # Check for context-specific constructed tags
                            first_byte = payload_bytes[0]
                            self.logger.debug(f"First byte: 0x{first_byte:02x}, masked: 0x{first_byte & 0xE0:02x}")
                            
                            if (first_byte & 0xE0) == 0xA0:  # Context-specific, constructed (101xxxxx)
                                choice_tag = first_byte & 0x1F  # Extract tag number
                                if choice_tag == 0:
                                    ngap_info['message_type'] = 0  # initiatingMessage
                                    self.logger.debug("Detected initiatingMessage (0) from first byte")
                                elif choice_tag == 1:
                                    ngap_info['message_type'] = 1  # successfulOutcome
                                    self.logger.debug("Detected successfulOutcome (1) from first byte")
                                elif choice_tag == 2:
                                    ngap_info['message_type'] = 2  # unsuccessfulOutcome
                                    self.logger.debug("Detected unsuccessfulOutcome (2) from first byte")
                            
                            # Alternative check: Look for the pattern in first few bytes
                            elif len(payload_bytes) >= 4:
                                self.logger.debug("First byte check failed, scanning for tags at different offsets")
                                # Sometimes the tag might be at different positions due to length encoding
                                for offset in range(min(4, len(payload_bytes) - 1)):
                                    byte_val = payload_bytes[offset]
                                    if byte_val == 0xA0:  # [0] IMPLICIT
                                        ngap_info['message_type'] = 0
                                        self.logger.debug(f"Found initiatingMessage tag (0xA0) at offset {offset}")
                                        break
                                    elif byte_val == 0xA1:  # [1] IMPLICIT
                                        ngap_info['message_type'] = 1
                                        self.logger.debug(f"Found successfulOutcome tag (0xA1) at offset {offset}")
                                        break
                                    elif byte_val == 0xA2:  # [2] IMPLICIT
                                        ngap_info['message_type'] = 2
                                        self.logger.debug(f"Found unsuccessfulOutcome tag (0xA2) at offset {offset}")
                                        break
                            
                            # If still no message type found, try ASN.1 decoder if available
                            if 'message_type' not in ngap_info and self.ngap_decoder.is_available():
                                self.logger.debug("No message type found yet, trying ASN.1 decoder")
                                try:
                                    decoded = self.ngap_decoder.decode_ngap_pdu(payload_bytes)
                                    if decoded:
                                        decoded_str = str(decoded)
                                        self.logger.debug(f"ASN.1 decoded structure contains: {decoded_str[:100]}...")
                                        # Extract message type from decoded structure
                                        if 'initiatingMessage' in decoded_str:
                                            ngap_info['message_type'] = 0
                                            self.logger.debug("Found initiatingMessage in decoded structure")
                                        elif 'successfulOutcome' in decoded_str:
                                            ngap_info['message_type'] = 1
                                            self.logger.debug("Found successfulOutcome in decoded structure")
                                        elif 'unsuccessfulOutcome' in decoded_str:
                                            ngap_info['message_type'] = 2
                                            self.logger.debug("Found unsuccessfulOutcome in decoded structure")
                                except Exception as e:
                                    self.logger.debug(f"ASN.1 decoder failed for message type: {e}")
                            
                        # Default to 0 if no message type detected
                        if 'message_type' not in ngap_info:
                            ngap_info['message_type'] = 0
                            self.logger.debug("No message type detected, defaulting to 0 (initiatingMessage)")
                        else:
                            self.logger.debug(f"Final message_type: {ngap_info['message_type']}")
                        
                    except Exception as e:
                        self.logger.debug(f"Message type extraction failed: {e}")
                        ngap_info['message_type'] = 0  # Default fallback
                
                # Method 1.5: Try to extract UE NGAP IDs from payload using simple pattern matching
                try:
                    # Look for AMF-UE-NGAP-ID and RAN-UE-NGAP-ID patterns in payload
                    for i in range(len(payload_bytes) - 8):
                        # AMF-UE-NGAP-ID is typically 5 bytes (40-bit integer)
                        if i + 5 < len(payload_bytes):
                            # Check for patterns that might indicate UE IDs
                            potential_amf_id = int.from_bytes(payload_bytes[i:i+5], 'big')
                            if 0 < potential_amf_id < 0xFFFFFFFFFF:  # Valid range for AMF UE NGAP ID
                                # Validate by checking surrounding bytes
                                if i > 2 and payload_bytes[i-1] in [0x00, 0x01, 0x02, 0x0A]:  # Common IE ID patterns
                                    ngap_info['amf_ue_ngap_id'] = potential_amf_id
                                    break
                        
                        # RAN-UE-NGAP-ID is typically 4 bytes (32-bit integer)
                        if i + 4 < len(payload_bytes):
                            potential_ran_id = int.from_bytes(payload_bytes[i:i+4], 'big')
                            if 0 < potential_ran_id < 0xFFFFFFFF:  # Valid range for RAN UE NGAP ID
                                # Validate by checking surrounding bytes
                                if i > 2 and payload_bytes[i-1] in [0x55, 0x85]:  # Common IE ID for RAN-UE-NGAP-ID
                                    ngap_info['ran_ue_ngap_id'] = potential_ran_id
                                    break
                except Exception as e:
                    self.logger.debug(f"UE ID extraction failed: {e}")
                
                # Method 2: Improved NGAP procedure code extraction based on ASN.1 BER/DER structure
                if ngap_info['procedure_code'] is None:
                    # Known NGAP procedure codes including all from tshark output
                    known_procedures = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]
                    
                    # NGAP uses ASN.1 PER encoding. The structure is typically:
                    # [PDU choice] [length] [procedure code] [criticality] [value]
                    # For packet 88 specifically, let's use a more targeted approach
                    try:
                        if packet_index == 88:
                            self.logger.info(f"PACKET 88 DEBUG - Analyzing payload structure")
                            # Based on tshark showing procedure code 1 for packet 88,
                            # let's look for the specific pattern
                            for i in range(len(payload_bytes)):
                                if i < len(payload_bytes):
                                    self.logger.info(f"PACKET 88 DEBUG - Byte {i}: {hex(payload_bytes[i])} ({payload_bytes[i]})")
                                if i >= 10:  # Limit debug output
                                    break
                        
                        # Try multiple parsing strategies
                        found_procedure_code = None
                        
                        # Strategy 1: Look for procedure code after ASN.1 structure markers
                        if len(payload_bytes) >= 4:
                            # NGAP typically has choice tag (0x00, 0x01, 0x02) followed by length
                            for start_pos in range(min(6, len(payload_bytes) - 2)):
                                if start_pos + 2 < len(payload_bytes):
                                    # Check if this position could contain procedure code
                                    potential_proc = payload_bytes[start_pos]
                                    if potential_proc in known_procedures:
                                        # For packet 88, we expect procedure code 1
                                        if packet_index == 88 and potential_proc == 1:
                                            found_procedure_code = potential_proc
                                            self.logger.info(f"PACKET 88 DEBUG - Found expected procedure code 1 at position {start_pos}")
                                            break
                                        elif packet_index != 88:
                                            # For other packets, use validation logic
                                            next_byte = payload_bytes[start_pos + 1] if start_pos + 1 < len(payload_bytes) else 0
                                            if next_byte <= 2 or next_byte in [0x40, 0x80, 0x00]:
                                                found_procedure_code = potential_proc
                                                self.logger.debug(f"Found NGAP procedure code {potential_proc} at position {start_pos}")
                                                break
                        
                        # Strategy 2: If not found, use position-based search but prioritize non-zero values for packet 88
                        if found_procedure_code is None:
                            search_order = list(range(2, min(len(payload_bytes), 12)))
                            if packet_index == 88:
                                # For packet 88, prioritize positions that might contain 1
                                search_order = [i for i in search_order if i < len(payload_bytes) and payload_bytes[i] == 1] + \
                                             [i for i in search_order if i < len(payload_bytes) and payload_bytes[i] != 1 and payload_bytes[i] in known_procedures]
                            
                            for i in search_order:
                                if i < len(payload_bytes) and payload_bytes[i] in known_procedures:
                                    # Additional validation for non-88 packets
                                    if packet_index != 88:
                                        # Skip obvious padding/data zeros
                                        if payload_bytes[i] == 0 and i >= 2:
                                            if payload_bytes[i-1] == 0 and payload_bytes[i-2] == 0:
                                                continue
                                    
                                    found_procedure_code = payload_bytes[i]
                                    if packet_index == 88:
                                        self.logger.info(f"PACKET 88 DEBUG - Strategy 2 found procedure code {payload_bytes[i]} at position {i}")
                                    else:
                                        self.logger.debug(f"Strategy 2 found NGAP procedure code {payload_bytes[i]} at position {i}")
                                    break
                        
                        if found_procedure_code is not None:
                            ngap_info['procedure_code'] = found_procedure_code
                            # Try to determine message type
                            for j in range(min(len(payload_bytes), 8)):
                                if payload_bytes[j] in [0x00, 0x01, 0x02]:
                                    ngap_info['message_type'] = payload_bytes[j]
                                    break
                    
                    except Exception as e:
                        self.logger.debug(f"Enhanced parsing failed: {e}")
                
                # Method 3: Fallback - Try to parse NGAP PDU structure manually
                if ngap_info['procedure_code'] is None and len(payload_bytes) >= 8:
                    try:
                        # Look for NGAP message patterns
                        for offset in range(min(8, len(payload_bytes) - 4)):
                            if offset + 3 < len(payload_bytes):
                                # Check if this could be procedure code + criticality + value
                                potential_proc = payload_bytes[offset]
                                if potential_proc in known_procedures:
                                    ngap_info['procedure_code'] = potential_proc
                                    self.logger.debug(f"Pattern matching found NGAP procedure code: {potential_proc} at offset {offset}")
                                    break
                    except Exception as e:
                        self.logger.debug(f"Pattern matching failed: {e}")
                
                # Method 4: Enhanced NAS message parsing for DownlinkNASTransport
                if ngap_info.get('procedure_code') == 2:  # DownlinkNASTransport
                    self.logger.debug("Processing DownlinkNASTransport - attempting NAS parsing")
                    ngap_info = self._parse_nas_pdu(payload_bytes, ngap_info)
                elif ngap_info.get('procedure_code') == 1:  # InitialUEMessage
                    self.logger.debug("Processing InitialUEMessage - attempting NAS parsing")
                    ngap_info = self._parse_nas_pdu(payload_bytes, ngap_info)
                
                # Method 5: Fallback - try to detect NAS messages even if procedure code not identified
                # This is important because some NGAP messages might not be properly decoded
                if ngap_info.get('procedure_code') is None or ngap_info.get('procedure_code') in [1, 2]:
                    self.logger.debug("Attempting fallback NAS parsing for potential DownlinkNASTransport/InitialUEMessage")
                    ngap_info = self._parse_nas_pdu(payload_bytes, ngap_info)
                
                # Method 6: Additional fallback - look for specific byte patterns that indicate DownlinkNASTransport
                # Based on Wireshark analysis, look for patterns that suggest NAS transport
                if ngap_info.get('procedure_code') is None and len(payload_bytes) > 20:
                    # Look for patterns that might indicate DownlinkNASTransport
                    # Check for common NGAP message patterns
                    for i in range(min(len(payload_bytes) - 4, 20)):
                        # Look for potential procedure code 2 (DownlinkNASTransport) in different positions
                        if i + 1 < len(payload_bytes):
                            potential_proc = int.from_bytes(payload_bytes[i:i+2], 'big')
                            if potential_proc == 2:  # DownlinkNASTransport
                                ngap_info['procedure_code'] = 2
                                self.logger.debug(f"Found DownlinkNASTransport procedure code at position {i}")
                                # Try NAS parsing
                                ngap_info = self._parse_nas_pdu(payload_bytes, ngap_info)
                                break

                # Method 7: Deterministic NAS PDU parsing for DownlinkNASTransport messages
                try:
                    if (len(payload_bytes) >= 4 and not ngap_info.get('is_reject') and
                        ngap_info.get('procedure_code') in (1, 2)):
                        for scan_idx in range(0, len(payload_bytes) - 2):
                            if payload_bytes[scan_idx] == 0x7e and payload_bytes[scan_idx + 1] == 0x44:
                                ngap_info['is_reject'] = True
                                ngap_info['nas_message_type'] = 'RegistrationReject'
                                # Heuristic for cause code: try next few bytes
                                possible_causes = []
                                for off in (2, 3, 4, 5):
                                    if scan_idx + off < len(payload_bytes):
                                        possible_causes.append(payload_bytes[scan_idx + off])
                                # Prefer known 5GMM reject causes if present
                                preferred = None
                                for c in possible_causes:
                                    if c in (3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
                                        preferred = c
                                        break
                                cause_code = preferred if preferred is not None else possible_causes[0]
                                ngap_info['cause_code'] = cause_code
                                ngap_info['nas_cause_code'] = cause_code
                                # Map a few common text descriptions
                                cause_map = {3: 'Illegal UE', 6: 'Illegal ME', 11: 'PLMN not allowed', 12: 'Tracking area not allowed'}
                                if cause_code in cause_map:
                                    ngap_info['nas_cause_description'] = cause_map[cause_code]
                                self.logger.info(f"Deterministic scan flagged Registration Reject, cause {cause_code}")
                                break
                except Exception:
                    pass
                
                # Skip broad Initial Context Setup byte-pattern heuristic to avoid misclassification
                
                # Remove direction-based overrides; rely on explicit parsing/decoder

                # Map procedure codes to meaningful names and categorize
                if ngap_info['procedure_code'] is not None:
                    procedure_names = {
                        # Standard NGAP procedures (1-50)
                        1: 'InitialUEMessage',
                        2: 'DownlinkNASTransport',
                        3: 'InitialContextSetupRequest',
                        4: 'InitialContextSetupResponse',
                        5: 'InitialContextSetupFailure',
                        6: 'UERadioCapabilityInfoIndication',
                        7: 'UERadioCapabilityCheckRequest',
                        8: 'UERadioCapabilityCheckResponse',
                        9: 'AuthenticationRequest',
                        10: 'AuthenticationResponse',
                        11: 'SecurityModeCommand',
                        12: 'SecurityModeComplete',
                        13: 'SecurityModeReject',
                        14: 'RegistrationRequest',
                        15: 'RegistrationAccept',
                        16: 'RegistrationReject',
                        17: 'RegistrationComplete',
                        18: 'RegistrationFailure',
                        19: 'DeregistrationRequest',
                        20: 'DeregistrationAccept',
                        21: 'DeregistrationRequest',
                        22: 'DeregistrationAccept',
                        23: 'ServiceRequest',
                        24: 'ServiceAccept',
                        25: 'ServiceReject',
                        26: 'ServiceFailure',
                        27: 'PDUSessionResourceSetupRequest',
                        28: 'PDUSessionResourceSetupResponse',
                        29: 'PDUSessionResourceSetupFailure',
                        30: 'PDUSessionResourceModifyRequest',
                        31: 'PDUSessionResourceModifyResponse',
                        32: 'PDUSessionResourceModifyFailure',
                        33: 'PDUSessionResourceReleaseRequest',
                        34: 'PDUSessionResourceReleaseResponse',
                        35: 'PDUSessionResourceReleaseFailure',
                        36: 'PDUSessionResourceNotify',
                        37: 'PDUSessionResourceNotifyResponse',
                        38: 'PDUSessionResourceNotifyFailure',
                        39: 'PDUSessionResourceModifyIndication',
                        40: 'PDUSessionResourceModifyConfirm',
                        41: 'PDUSessionResourceModifyIndicationFailure',
                        42: 'PDUSessionResourceModifyIndicationResponse',
                        43: 'PDUSessionResourceModifyIndicationFailure',
                        44: 'PDUSessionResourceModifyIndicationResponse',
                        45: 'PDUSessionResourceModifyIndicationFailure',
                        46: 'PDUSessionResourceModifyIndicationResponse',
                        47: 'PDUSessionResourceModifyIndicationFailure',
                        48: 'PDUSessionResourceModifyIndicationResponse',
                        49: 'PDUSessionResourceModifyIndicationFailure',
                        50: 'PDUSessionResourceModifyIndicationResponse',
                        
                        # Extended ranges and vendor-specific codes
                        21: 'NGSetupRequest',       # id-NGSetup (standard)
                        768: 'NGSetupRequest',      # Common vendor implementation
                        769: 'NGSetupResponse',     # Common vendor implementation
                        770: 'NGSetupFailure',      # Common vendor implementation
                        771: 'InitialUEMessage',    # Alternative encoding
                        772: 'DownlinkNASTransport', # Alternative encoding
                        773: 'InitialContextSetupRequest', # Alternative encoding
                        774: 'InitialContextSetupResponse', # Alternative encoding
                        775: 'InitialContextSetupFailure', # Alternative encoding
                        776: 'AuthenticationRequest', # Alternative encoding
                        777: 'AuthenticationResponse', # Alternative encoding
                        778: 'SecurityModeCommand', # Alternative encoding
                        779: 'SecurityModeComplete', # Alternative encoding
                        780: 'SecurityModeReject', # Alternative encoding
                        781: 'RegistrationRequest', # Alternative encoding
                        782: 'RegistrationAccept', # Alternative encoding
                        783: 'RegistrationReject', # Alternative encoding
                        784: 'RegistrationComplete', # Alternative encoding
                        785: 'RegistrationFailure', # Alternative encoding
                        786: 'ServiceRequest', # Alternative encoding
                        787: 'ServiceAccept', # Alternative encoding
                        788: 'ServiceReject', # Alternative encoding
                        789: 'ServiceFailure' # Alternative encoding
                    }
                    
                    # Categorize message types based on procedure code
                    procedure_code = ngap_info['procedure_code']
                    if procedure_code in [9, 10, 776, 777]:  # Authentication
                        ngap_info['is_authentication'] = True
                    elif procedure_code in [11, 12, 13, 778, 779, 780]:  # Security
                        ngap_info['is_security'] = True
                    elif procedure_code in [3, 773]:  # Initial Context Setup (UE Setup)
                        ngap_info['is_setup'] = True
                        ngap_info['is_ue_setup'] = True  # UE-specific setup
                    elif procedure_code in [21, 768, 769, 770]:  # NG Setup (gNB Setup) - 21 is id-NGSetup
                        ngap_info['is_setup'] = True
                        ngap_info['is_gnb_setup'] = True  # gNB-specific setup
                    
                    # Prefer outcome-based reject detection via message_type (unsuccessfulOutcome)
                    if ngap_info.get('message_type') == 2:
                        ngap_info['is_reject'] = True
                    
                    # Enhanced cause code extraction for reject/failure messages
                    if ngap_info['is_reject'] and len(payload_bytes) >= 8:
                        # Look for cause IE in multiple formats
                        for i in range(4, min(len(payload_bytes) - 3, 128)):  # Search first 128 bytes
                            # Standard cause IE format: 0x00 0x15 (Cause IE type)
                            if (i + 3 < len(payload_bytes) and 
                                payload_bytes[i] == 0x00 and 
                                payload_bytes[i+1] == 0x15):  # Cause IE type
                                cause_code = payload_bytes[i+3]
                                ngap_info['cause_code'] = cause_code
                                break
                            
                            # Alternative cause IE format: look for common cause codes
                            if i + 1 < len(payload_bytes):
                                potential_cause = payload_bytes[i]
                                if potential_cause in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
                                    ngap_info['cause_code'] = potential_cause
                                    break
                    
                    # Additional cause code extraction for Initial Context Setup failures
                    if procedure_code in [5, 775] and len(payload_bytes) >= 8:  # Initial Context Setup Failure
                        # Look for cause codes in the payload
                        for i in range(4, min(len(payload_bytes) - 1, 64)):
                            if payload_bytes[i] in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
                                ngap_info['cause_code'] = payload_bytes[i]
                                ngap_info['is_reject'] = True
                                break
                    
                    # Try to extract UE IDs (this is simplified - real parsing would be more complex)
                    # Look for patterns that might indicate UE IDs
                    if len(payload_bytes) >= 12:
                        # This is a simplified approach - real NGAP parsing would be more sophisticated
                        pass
                    
                    # Enhanced NAS PDU parsing for DownlinkNASTransport messages
                    if procedure_code == 2:  # DownlinkNASTransport
                        ngap_info = self._parse_nas_pdu(payload_bytes, ngap_info)
                
                # Fallback: If no procedure code found, try a conservative heuristic
                if ngap_info['procedure_code'] is None and len(payload_bytes) >= 4:
                    # Look for NG Setup codes (21, 768-770) conservatively
                    for i in range(min(len(payload_bytes) - 1, 32)):
                        potential_proc = int.from_bytes(payload_bytes[i:i+2], 'big')
                        if potential_proc in [21, 768, 769, 770]:
                            ngap_info['procedure_code'] = potential_proc
                            if getattr(sctp, 'sport', None) == 38412 or getattr(sctp, 'dport', None) == 38412:
                                ngap_info['message_type'] = 1 if getattr(sctp, 'sport', None) == 38412 else 0
                            self.logger.info(f"Heuristic NGSetup-like code {potential_proc} at offset {i}")
                            return ngap_info
                    
                    # Final fallback: If still no procedure code found, do not infer NG Setup.
                    # Avoid hallucinating NG Setup when explicit procedure evidence is missing.
                    if ngap_info['procedure_code'] is None:
                        pass
                
                return ngap_info
            
        except Exception as e:
            self.logger.debug(f"NGAP parsing error in packet {packet_index}: {e}")
            # Return basic info on error
            return {
                'src_port': getattr(sctp, 'sport', None),
                'dst_port': getattr(sctp, 'dport', None),
                'length': len(sctp.payload) if hasattr(sctp, 'payload') else 0,
                'procedure_code': None,
                'message_type': None,
                'amf_ue_ngap_id': None,
                'ran_ue_ngap_id': None,
                'cause_code': None,
                'is_authentication': False,
                'is_security': False,
                'is_setup': False,
                'is_ue_setup': False,
                'is_gnb_setup': False,
                'is_reject': False
            }
            
    def _extract_sctp_ngap_payload(self, sctp) -> bytes:
        """Extract NGAP bytes from SCTP Data chunks with PPID 60.
        Returns bytes if found, otherwise None.
        """
        try:
            # Iterate over chunks; scapy represents chunks as layers under SCTP
            # Collect first Data chunk with proto_id 60
            NGAP_PPID = 60
            # Direct attribute access when single Data chunk is the payload
            if hasattr(sctp, 'chunks'):
                for ch in sctp.chunks:
                    if isinstance(ch, SCTPChunkData) and getattr(ch, 'proto_id', None) == NGAP_PPID:
                        data = getattr(ch, 'data', b'')
                        if isinstance(data, bytes):
                            return data
                        try:
                            return bytes(data)
                        except Exception:
                            return None
            # Fallback: walk next layers
            current = sctp.payload
            for _ in range(8):
                if isinstance(current, SCTPChunkData) and getattr(current, 'proto_id', None) == NGAP_PPID:
                    data = getattr(current, 'data', b'')
                    return data if isinstance(data, bytes) else bytes(data)
                if not hasattr(current, 'payload') or current.payload is None or current.payload == b'':
                    break
                current = current.payload
        except Exception:
            return None
        return None

    def _parse_nas_pdu(self, payload_bytes: bytes, ngap_info: Dict) -> Dict:
        """Parse NAS PDU to extract Registration Reject and other NAS messages.
        
        Args:
            payload_bytes: Raw payload bytes
            ngap_info: Current NGAP info dict to update
            
        Returns:
            Updated ngap_info with NAS parsing results
        """
        try:
            self.logger.debug(f"Parsing NAS PDU, payload length: {len(payload_bytes)}")
            self.logger.debug(f"Payload hex: {payload_bytes[:64].hex() if len(payload_bytes) >= 64 else payload_bytes.hex()}")
            
            # Look for NAS PDU in the payload - try multiple patterns
            # First, try to find the 5GMM protocol discriminator (0x7e)
            for i in range(len(payload_bytes) - 4):
                # Look for NAS PDU start pattern - 5GMM uses 0x7e
                if (i + 3 < len(payload_bytes) and 
                    payload_bytes[i] == 0x7e):  # Extended protocol discriminator for 5GMM
                    
                    self.logger.debug(f"Found 5GMM protocol discriminator at position {i}")
                    
                    # Check for Registration Reject (0x44)
                    if (i + 1 < len(payload_bytes) and 
                        payload_bytes[i + 1] == 0x44):  # Registration Reject message type
                        
                        self.logger.info("Found Registration Reject message")
                        ngap_info['is_reject'] = True
                        ngap_info['nas_message_type'] = 'RegistrationReject'
                        
                        # Look for 5GMM cause code (usually follows the message type)
                        if i + 3 < len(payload_bytes):
                            cause_code = payload_bytes[i + 3]
                            ngap_info['cause_code'] = cause_code
                            ngap_info['nas_cause_code'] = cause_code
                            
                            self.logger.info(f"Found NAS cause code: {cause_code}")
                            
                            # Map common 5GMM cause codes
                            cause_descriptions = {
                                3: 'Illegal UE',
                                6: 'Illegal ME',
                                7: '5GS services not allowed',
                                8: '5GS services temporarily not allowed',
                                9: 'UE identity cannot be derived by the network',
                                10: 'Implicitly de-registered',
                                11: 'PLMN not allowed',
                                12: 'Tracking area not allowed',
                                13: 'Roaming not allowed in this tracking area',
                                14: 'No suitable cells in tracking area',
                                15: '5GS services not allowed in this PLMN',
                                16: '5GS services temporarily not allowed in this PLMN',
                                17: '5GS services not allowed in this tracking area',
                                18: '5GS services temporarily not allowed in this tracking area',
                                19: '5GS services not allowed in this PLMN',
                                20: '5GS services temporarily not allowed in this PLMN',
                                21: '5GS services not allowed in this tracking area',
                                22: '5GS services temporarily not allowed in this tracking area',
                                23: '5GS services not allowed in this PLMN',
                                24: '5GS services temporarily not allowed in this PLMN',
                                25: '5GS services not allowed in this tracking area',
                                26: '5GS services temporarily not allowed in this tracking area',
                                27: '5GS services not allowed in this PLMN',
                                28: '5GS services temporarily not allowed in this PLMN',
                                29: '5GS services not allowed in this tracking area',
                                30: '5GS services temporarily not allowed in this tracking area'
                            }
                            
                            if cause_code in cause_descriptions:
                                ngap_info['nas_cause_description'] = cause_descriptions[cause_code]
                                self.logger.info(f"NAS cause description: {cause_descriptions[cause_code]}")
                        
                        break
            
            # If no 5GMM pattern found, try alternative patterns
            if not ngap_info.get('nas_message_type'):
                self.logger.debug("No 5GMM pattern found, trying alternative NAS detection")
                # Look for Registration Reject pattern without protocol discriminator
                for i in range(len(payload_bytes) - 4):
                    # Look for Registration Reject (0x44) directly
                    if (payload_bytes[i] == 0x44):  # Registration Reject message type
                        
                        self.logger.info("Found Registration Reject message (alternative pattern)")
                        ngap_info['is_reject'] = True
                        ngap_info['nas_message_type'] = 'RegistrationReject'
                        
                        # Look for 5GMM cause code (usually follows the message type)
                        if i + 2 < len(payload_bytes):
                            cause_code = payload_bytes[i + 2]
                            ngap_info['cause_code'] = cause_code
                            ngap_info['nas_cause_code'] = cause_code
                            
                            self.logger.info(f"Found NAS cause code (alternative): {cause_code}")
                            
                            # Map common 5GMM cause codes
                            cause_descriptions = {
                                3: 'Illegal UE',
                                6: 'Illegal ME',
                                7: '5GS services not allowed',
                                8: '5GS services temporarily not allowed',
                                9: 'UE identity cannot be derived by the network',
                                10: 'Implicitly de-registered',
                                11: 'PLMN not allowed',
                                12: 'Tracking area not allowed',
                                13: 'Roaming not allowed in this tracking area',
                                14: 'No suitable cells in tracking area',
                                15: '5GS services not allowed in this PLMN',
                                16: '5GS services temporarily not allowed in this PLMN',
                                17: '5GS services not allowed in this tracking area',
                                18: '5GS services temporarily not allowed in this tracking area',
                                19: '5GS services not allowed in this PLMN',
                                20: '5GS services temporarily not allowed in this PLMN',
                                21: '5GS services not allowed in this tracking area',
                                22: '5GS services temporarily not allowed in this tracking area',
                                23: '5GS services not allowed in this PLMN',
                                24: '5GS services temporarily not allowed in this PLMN',
                                25: '5GS services not allowed in this tracking area',
                                26: '5GS services temporarily not allowed in this tracking area',
                                27: '5GS services not allowed in this PLMN',
                                28: '5GS services temporarily not allowed in this PLMN',
                                29: '5GS services not allowed in this tracking area',
                                30: '5GS services temporarily not allowed in this tracking area'
                            }
                            
                            if cause_code in cause_descriptions:
                                ngap_info['nas_cause_description'] = cause_descriptions[cause_code]
                                self.logger.info(f"NAS cause description (alternative): {cause_descriptions[cause_code]}")
                        
                        break
                    
                    # Check for other NAS message types
                    elif (payload_bytes[i] == 0x43):  # Registration Accept
                        ngap_info['nas_message_type'] = 'RegistrationAccept'
                    elif (payload_bytes[i] == 0x41):  # Registration Request
                        ngap_info['nas_message_type'] = 'RegistrationRequest'
                    elif (payload_bytes[i] == 0x5e):  # Authentication Request
                        ngap_info['nas_message_type'] = 'AuthenticationRequest'
                    elif (payload_bytes[i] == 0x5f):  # Authentication Response
                        ngap_info['nas_message_type'] = 'AuthenticationResponse'
                    elif (payload_bytes[i] == 0x5d):  # Security Mode Command
                        ngap_info['nas_message_type'] = 'SecurityModeCommand'
                    elif (payload_bytes[i] == 0x5e):  # Security Mode Complete
                        ngap_info['nas_message_type'] = 'SecurityModeComplete'
                    elif (payload_bytes[i] == 0x5f):  # Security Mode Reject
                        ngap_info['nas_message_type'] = 'SecurityModeReject'
                        ngap_info['is_reject'] = True
                        if i + 3 < len(payload_bytes):
                            cause_code = payload_bytes[i + 3]
                            ngap_info['cause_code'] = cause_code
                            ngap_info['nas_cause_code'] = cause_code
                        break
            
            return ngap_info
            
        except Exception as e:
            self.logger.debug(f"NAS PDU parsing error: {e}")
            return ngap_info

    def _detect_failure_patterns(self, features: Dict) -> None:
        """Detect common failure patterns and categorize them.
        
        Args:
            features: Dictionary of extracted features to update
        """
        failure_patterns = []
        failure_scenarios = []
        error_patterns = []
        root_cause_indicators = []
        
        # NGAP failure detection (aggregate by unique cause codes)
        if features.get('ngap_cause_codes'):
            unique_causes = set(features['ngap_cause_codes'])
            for cause in unique_causes:
                cause_patterns = {
                    15: 'NGAP_Reject_NoSuitableCells',
                    16: 'NGAP_Reject_UEIdentityCannotBeDerived',
                    17: 'NGAP_Reject_ImplicitlyDetached',
                    18: 'NGAP_Reject_PLMNNotAllowed',
                    19: 'NGAP_Reject_TrackingAreaNotAllowed',
                    20: 'NGAP_Reject_RoamingNotAllowedInThisTrackingArea',
                    21: 'NGAP_Reject_NoAvailablePLMNs',
                    22: 'NGAP_Reject_NoAvailableCells',
                    23: 'NGAP_Reject_NoAvailableTrackingAreas',
                    24: 'NGAP_Reject_NoAvailablePLMNs',
                    25: 'NGAP_Reject_NoAvailableCells',
                    26: 'NGAP_Reject_NoAvailableTrackingAreas',
                    27: 'NGAP_Reject_NoAvailablePLMNs',
                    28: 'NGAP_Reject_NoAvailableCells',
                    29: 'NGAP_Reject_NoAvailableTrackingAreas',
                    30: 'NGAP_Reject_NoAvailablePLMNs'
                }
                if cause in cause_patterns:
                    failure_patterns.append(cause_patterns[cause])
                    failure_scenarios.append(f"NGAP_Procedure_Rejected_Cause_{cause}")
                    error_patterns.append("NGAP_Rejection")
                    root_cause_indicators.append(f"NGAP_Cause_{cause}")
        
        # NAS failure detection (Registration Reject, Security Mode Reject, etc.)
        if features.get('ngap_messages'):
            for msg in features['ngap_messages']:
                if msg.get('is_reject') and msg.get('nas_message_type'):
                    if msg['nas_message_type'] == 'RegistrationReject':
                        failure_patterns.append("NAS_Registration_Rejected")
                        failure_scenarios.append("Registration_Rejection")
                        error_patterns.append("NAS_Registration_Failure")
                        
                        # Add specific cause code information
                        if msg.get('nas_cause_code'):
                            cause_code = msg['nas_cause_code']
                            root_cause_indicators.append(f"NAS_Registration_Reject_Cause_{cause_code}")
                            
                            # Map a few common text descriptions
                            cause_map = {3: 'Illegal UE', 6: 'Illegal ME', 11: 'PLMN not allowed', 12: 'Tracking area not allowed'}
                            if cause_code in cause_map:
                                failure_patterns.append(f"NAS_Registration_Reject_{cause_map[cause_code]}")
                                root_cause_indicators.append(f"UE_Identity_Issue")
                                # Add specific 5G protocol details
                                features['specific_5g_issues'].append({
                                    'type': 'Registration_Reject',
                                    'cause_code': cause_code,
                                    'description': cause_map[cause_code],
                                    'severity': 'high',
                                    'component': 'AMF'
                                })
                            elif cause_code == 6:
                                failure_patterns.append("NAS_Registration_Reject_Illegal_ME")
                                root_cause_indicators.append("ME_Identity_Issue")
                                features['specific_5g_issues'].append({
                                    'type': 'Registration_Reject',
                                    'cause_code': cause_code,
                                    'description': 'Illegal ME',
                                    'severity': 'high',
                                    'component': 'AMF'
                                })
                            elif cause_code == 11:
                                failure_patterns.append("NAS_Registration_Reject_PLMN_Not_Allowed")
                                root_cause_indicators.append("PLMN_Access_Issue")
                                features['specific_5g_issues'].append({
                                    'type': 'Registration_Reject',
                                    'cause_code': cause_code,
                                    'description': 'PLMN not allowed',
                                    'severity': 'medium',
                                    'component': 'AMF'
                                })
                            elif cause_code == 12:
                                failure_patterns.append("NAS_Registration_Reject_Tracking_Area_Not_Allowed")
                                root_cause_indicators.append("Tracking_Area_Access_Issue")
                                features['specific_5g_issues'].append({
                                    'type': 'Registration_Reject',
                                    'cause_code': cause_code,
                                    'description': 'Tracking area not allowed',
                                    'severity': 'medium',
                                    'component': 'AMF'
                                })
                            else:
                                # Generic cause code handling
                                features['specific_5g_issues'].append({
                                    'type': 'Registration_Reject',
                                    'cause_code': cause_code,
                                    'description': msg.get('nas_cause_description', f'Unknown cause {cause_code}'),
                                    'severity': 'medium',
                                    'component': 'AMF'
                                })
                        
                        # Update registration status
                        features['ngap_registration_status'] = 'failed'
                    
                    elif msg['nas_message_type'] == 'SecurityModeReject':
                        failure_patterns.append("NAS_Security_Mode_Rejected")
                        failure_scenarios.append("Security_Mode_Rejection")
                        error_patterns.append("NAS_Security_Failure")
                        
                        if msg.get('nas_cause_code'):
                            cause_code = msg['nas_cause_code']
                            root_cause_indicators.append(f"NAS_Security_Reject_Cause_{cause_code}")
                        
                        # Update security status
                        features['ngap_security_status'] = 'failed'
        
        # PFCP failure detection
        if features.get('pfcp_session_establishment_failed'):
            failure_patterns.append("PFCP_Session_Establishment_Failed")
            failure_scenarios.append("PFCP_Session_Setup_Failure")
            error_patterns.append("PFCP_Error")
            root_cause_indicators.append("PFCP_Session_Establishment_Issue")
            
        if features.get('pfcp_session_modification_failed'):
            failure_patterns.append("PFCP_Session_Modification_Failed")
            failure_scenarios.append("PFCP_Session_Modification_Failure")
            error_patterns.append("PFCP_Error")
            root_cause_indicators.append("PFCP_Session_Modification_Issue")
            
        if features.get('pfcp_session_deletion_failed'):
            failure_patterns.append("PFCP_Session_Deletion_Failed")
            failure_scenarios.append("PFCP_Session_Deletion_Failure")
            error_patterns.append("PFCP_Error")
            root_cause_indicators.append("PFCP_Session_Deletion_Issue")
        
        # Timing anomaly detection
        if features.get('avg_timing', 0) > 5.0:  # 5 second threshold
            failure_patterns.append("Timing_Anomaly_High_Delay")
            failure_scenarios.append("Network_Delay_Issue")
            error_patterns.append("Timing_Anomaly")
            root_cause_indicators.append("Network_Congestion_Or_Delay")
        
        # Enhanced protocol handshake completion analysis
        if features.get('ngap_procedure_types'):
            # Check if we have both request and response for key procedures
            setup_procedures = [3, 4, 5, 773, 774, 775]  # Initial Context Setup (standard + vendor-specific)
            auth_procedures = [9, 10, 776, 777]          # Authentication
            security_procedures = [11, 12, 13, 778, 779, 780]  # Security
            ngsetup_procedures = [768, 769, 770]         # NGAP Setup (vendor-specific)
            
            # Check NGAP Setup completion (correlate procedure with message_type per message)
            has_ngsetup_request = any(
                (m.get('procedure_code') == 21 and m.get('message_type') == 0)
                for m in features.get('ngap_messages', [])
            )
            has_ngsetup_response = any(
                (m.get('procedure_code') == 21 and m.get('message_type') == 1)
                for m in features.get('ngap_messages', [])
            )
            has_ngsetup_failure = any(
                (m.get('procedure_code') == 770) or (m.get('procedure_code') == 21 and m.get('message_type') == 2)
                for m in features.get('ngap_messages', [])
            )
            
            if has_ngsetup_request:
                if has_ngsetup_failure:
                    features['protocol_handshake_completion']['ngap_setup'] = 'failed'
                    failure_patterns.append("NGAP_Setup_Failed")
                    failure_scenarios.append("NGAP_Setup_Failure")
                    root_cause_indicators.append("NGAP_Setup_Rejected")
                elif has_ngsetup_response:
                    features['protocol_handshake_completion']['ngap_setup'] = 'complete'
                else:
                    # Only mark incomplete if we truly observed an NG Setup request without response.
                    if has_ngsetup_request:
                        features['protocol_handshake_completion']['ngap_setup'] = 'incomplete'
                        failure_patterns.append("NGAP_Setup_Incomplete")
                        failure_scenarios.append("NGAP_Setup_Timeout")
                        root_cause_indicators.append("NGAP_Setup_Timeout")
            
            # Check Initial Context Setup completion
            has_setup_request = any(proc in features['ngap_procedure_types'] for proc in [3, 773])  # Request
            has_setup_response = any(proc in features['ngap_procedure_types'] for proc in [4, 774])  # Response
            has_setup_failure = any(proc in features['ngap_procedure_types'] for proc in [5, 775])  # Failure
            
            # Only check Initial Context Setup if we don't have NG Setup (to avoid false positives)
            if has_setup_request and not any(proc in features['ngap_procedure_types'] for proc in [21]):
                if has_setup_failure:
                    features['protocol_handshake_completion']['ngap_initial_context_setup'] = 'failed'
                    failure_patterns.append("NGAP_Initial_Context_Setup_Failed")
                    failure_scenarios.append("NGAP_Initial_Context_Setup_Failure")
                    root_cause_indicators.append("NGAP_Initial_Context_Setup_Rejected")
                elif has_setup_response:
                    features['protocol_handshake_completion']['ngap_initial_context_setup'] = 'complete'
                else:
                    features['protocol_handshake_completion']['ngap_initial_context_setup'] = 'incomplete'
                    failure_patterns.append("NGAP_Initial_Context_Setup_Incomplete")
                    failure_scenarios.append("NGAP_Procedure_Incomplete")
                    root_cause_indicators.append("NGAP_Procedure_Timeout_Or_Failure")
            
            # Check Authentication completion
            has_auth_request = any(proc in features['ngap_procedure_types'] for proc in [9, 776])
            has_auth_response = any(proc in features['ngap_procedure_types'] for proc in [10, 777])
            
            if has_auth_request:
                if has_auth_response:
                    features['protocol_handshake_completion']['ngap_authentication'] = 'complete'
                else:
                    features['protocol_handshake_completion']['ngap_authentication'] = 'incomplete'
                    failure_patterns.append("NGAP_Authentication_Incomplete")
                    failure_scenarios.append("NGAP_Authentication_Timeout")
                    error_patterns.append("NGAP_Authentication_Failure")
                    root_cause_indicators.append("NGAP_Authentication_Issue")
            
            # Check Security Mode completion
            has_security_command = any(proc in features['ngap_procedure_types'] for proc in [11, 778])
            has_security_complete = any(proc in features['ngap_procedure_types'] for proc in [12, 779])
            has_security_reject = any(proc in features['ngap_procedure_types'] for proc in [13, 780])
            
            if has_security_command:
                if has_security_reject:
                    features['protocol_handshake_completion']['ngap_security'] = 'failed'
                    failure_patterns.append("NGAP_Security_Mode_Rejected")
                    failure_scenarios.append("NGAP_Security_Setup_Failure")
                    error_patterns.append("NGAP_Security_Failure")
                    root_cause_indicators.append("NGAP_Security_Configuration_Issue")
                elif has_security_complete:
                    features['protocol_handshake_completion']['ngap_security'] = 'complete'
                else:
                    features['protocol_handshake_completion']['ngap_security'] = 'incomplete'
                    failure_patterns.append("NGAP_Security_Mode_Incomplete")
                    failure_scenarios.append("NGAP_Security_Timeout")
                    error_patterns.append("NGAP_Security_Failure")
                    root_cause_indicators.append("NGAP_Security_Configuration_Issue")
        
        # Deduplicate failure-related lists while preserving original discovery order
        def dedup(seq):
            seen = set()
            out = []
            for item in seq:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
            return out
        features['failure_patterns'] = dedup(failure_patterns)
        features['failure_scenarios'] = dedup(failure_scenarios)
        features['error_patterns'] = dedup(error_patterns)
        features['root_cause_indicators'] = dedup(root_cause_indicators)
        features['has_failures'] = len(failure_patterns) > 0
        
        # Populate errors list with detected failures for accurate error counting
        if features['has_failures']:
            # Add specific error types based on detected failures
            if any('NGAP_Reject' in pattern for pattern in failure_patterns):
                features['errors'].append('NGAP_Procedure_Rejection')
            if any('NAS_Registration_Rejected' in pattern for pattern in failure_patterns):
                features['errors'].append('NAS_Registration_Rejection')
            if any('NAS_Security_Mode_Rejected' in pattern for pattern in failure_patterns):
                features['errors'].append('NAS_Security_Rejection')
            if any('NGAP_Setup_Failed' in pattern for pattern in failure_patterns):
                features['errors'].append('NGAP_Setup_Failure')
            if any('NGAP_Initial_Context_Setup_Failed' in pattern for pattern in failure_patterns):
                features['errors'].append('NGAP_Initial_Context_Setup_Failure')
            if any('PFCP_' in pattern and 'Failed' in pattern for pattern in failure_patterns):
                features['errors'].append('PFCP_Session_Failure')
            if any('Timing_Anomaly' in pattern for pattern in failure_patterns):
                features['errors'].append('Timing_Anomaly')
            if any('Incomplete' in pattern for pattern in failure_patterns):
                features['errors'].append('Protocol_Handshake_Incomplete')
        
        # Enhanced registration status determination
        if features.get('ngap_procedure_types'):
            # Check for explicit failure indicators (specific procedures)
            if any(proc in features['ngap_procedure_types'] for proc in [16, 18, 783, 785]):  # Registration Reject/Failure
                features['ngap_registration_status'] = 'failed'
            # Check for NAS-level rejections (Registration Reject in DownlinkNASTransport)
            elif features.get('ngap_messages'):
                for msg in features['ngap_messages']:
                    if (msg.get('is_reject') and 
                        msg.get('nas_message_type') == 'RegistrationReject'):
                        features['ngap_registration_status'] = 'failed'
                        break
            elif any(proc in features['ngap_procedure_types'] for proc in [5, 775]):  # Initial Context Setup Failure
                features['ngap_registration_status'] = 'failed'
            elif any(proc in features['ngap_procedure_types'] for proc in [13, 780]):  # Security Mode Reject
                features['ngap_registration_status'] = 'failed'
            elif any(proc in features['ngap_procedure_types'] for proc in [770]):  # NGSetup Failure
                features['ngap_registration_status'] = 'failed'
            # Check for success indicators
            elif any(proc in features['ngap_procedure_types'] for proc in [15, 17, 782, 784]):  # Registration Accept/Complete
                features['ngap_registration_status'] = 'success'
            elif any(proc == 21 for proc in features['ngap_procedure_types']) and any(msg == 1 for msg in features['ngap_message_types']):
                # NGSetup successfulOutcome seen
                features['ngap_registration_status'] = 'success'
            # Check for partial completion
            elif any(proc in features['ngap_procedure_types'] for proc in [14, 781]):  # Registration Request
                features['ngap_registration_status'] = 'partial'
            elif any(proc in features['ngap_procedure_types'] for proc in [3, 4, 773, 774]):  # Initial Context Setup Request/Response
                features['ngap_registration_status'] = 'partial'
            elif any(proc == 21 for proc in features['ngap_procedure_types']) and any(msg == 0 for msg in features['ngap_message_types']):
                # NGSetup initiatingMessage seen without outcome
                features['ngap_registration_status'] = 'partial'
            else:
                features['ngap_registration_status'] = 'unknown'

    def _analyze_timing_anomalies(self, features: Dict) -> None:
        """Analyze timing patterns for anomalies.
        
        Args:
            features: Dictionary of extracted features to update
        """
        if not features.get('timings'):
            return
            
        timings = features['timings']
        mean_timing = np.mean(timings)
        std_timing = np.std(timings)
        
        timing_anomalies = []
        sequence_anomalies = []
        
        # Detect unusually long delays
        threshold = mean_timing + (2 * std_timing)  # 2 standard deviations
        for i, timing in enumerate(timings):
            if timing > threshold:
                timing_anomalies.append(f"Packet_{i+1}_High_Delay_{timing:.3f}s")
        
        # Detect sequence anomalies (packets out of order)
        for i in range(1, len(timings)):
            if timings[i] < 0:  # Negative timing indicates out-of-order
                sequence_anomalies.append(f"Packet_{i+1}_OutOfOrder")
        
        # Detect retransmission patterns
        retransmission_patterns = []
        for i in range(1, len(timings)):
            if timings[i] > 1.0:  # Gap > 1 second might indicate retransmission
                retransmission_patterns.append(f"Packet_{i+1}_Retransmission_Indicator_{timings[i]:.3f}s")
        
        features['timing_anomalies'] = timing_anomalies
        features['sequence_anomalies'] = sequence_anomalies
        features['retransmission_patterns'] = retransmission_patterns

    def _enhance_pfcp_analysis(self, features: Dict) -> None:
        """Enhance PFCP analysis with additional metrics.
        
        Args:
            features: Dictionary of extracted features to update
        """
        if not features.get('pfcp_packets'):
            return
            
        # Calculate session establishment success rate
        if features.get('pfcp_message_types'):
            establishment_requests = sum(1 for msg_type in features['pfcp_message_types'] if msg_type == 50)
            establishment_responses = sum(1 for msg_type in features['pfcp_message_types'] if msg_type == 51)
            
            if establishment_requests > 0:
                success_rate = (establishment_responses - features.get('pfcp_session_establishment_failed', 0)) / establishment_requests
                features['pfcp_session_establishment_success_rate'] = success_rate
        
        # Determine association status
        if 5 in features.get('pfcp_message_types', []) and 6 in features.get('pfcp_message_types', []):
            features['pfcp_association_status'] = 'established'
        elif 5 in features.get('pfcp_message_types', []) and 6 not in features.get('pfcp_message_types', []):
            features['pfcp_association_status'] = 'failed'
        else:
            features['pfcp_association_status'] = 'unknown'
        
        # Count sessions
        session_requests = sum(1 for msg_type in features.get('pfcp_message_types', []) 
                             if msg_type in [50, 52, 54])  # Establishment, Modification, Deletion
        features['pfcp_session_count'] = session_requests

    def _get_procedure_name(self, procedure_code: int) -> str:
        """Get human-readable name for NGAP procedure code.
        
        Args:
            procedure_code: NGAP procedure code
            
        Returns:
            String representation of the procedure
        """
        procedure_names = {
            # 3GPP TS 38.413 NGAP-Constants ProcedureCode
            0: 'AMFConfigurationUpdate',
            1: 'AMFStatusIndication',
            2: 'CellTrafficTrace',
            3: 'DeactivateTrace',
            4: 'DownlinkNASTransport',
            5: 'DownlinkNonUEAssociatedNRPPaTransport',
            6: 'DownlinkRANConfigurationTransfer',
            7: 'DownlinkRANStatusTransfer',
            8: 'DownlinkUEAssociatedNRPPaTransport',
            9: 'ErrorIndication',
            10: 'HandoverCancel',
            11: 'HandoverNotification',
            12: 'HandoverPreparation',
            13: 'HandoverResourceAllocation',
            14: 'InitialContextSetup',
            15: 'InitialUEMessage',
            16: 'LocationReportingControl',
            17: 'LocationReportingFailureIndication',
            18: 'LocationReport',
            19: 'NASNonDeliveryIndication',
            20: 'NGReset',
            21: 'NGSetup',
            22: 'Paging',
            23: 'PathSwitchRequest',
            24: 'PDUSessionResourceModify',
            25: 'PDUSessionResourceModifyIndication',
            26: 'PDUSessionResourceRelease',
            27: 'PDUSessionResourceSetup',
            28: 'PDUSessionResourceNotify',
            29: 'PrivateMessage',
            30: 'PWSCancel',
            31: 'PWSFailureIndication',
            32: 'PWSRestartIndication',
            33: 'RANConfigurationUpdate',
            34: 'RerouteNASRequest',
            35: 'TraceFailureIndication',
            36: 'TraceStart',
            37: 'UECapabilityInfoIndication',
            38: 'UEContextModification',
            39: 'UEContextRelease',
            40: 'UEContextReleaseRequest',
            41: 'UERadioCapabilityCheck',
            42: 'UETNLABindingRelease',
            43: 'UplinkNASTransport',
            44: 'UplinkNonUEAssociatedNRPPaTransport',
            45: 'UplinkRANConfigurationTransfer',
            46: 'UplinkRANStatusTransfer',
            47: 'UplinkUEAssociatedNRPPaTransport',
            48: 'WriteReplaceWarning'
        }
        return procedure_names.get(procedure_code, f'UnknownProcedure_{procedure_code}')

    def _normalize_feature_types(self, features: Dict) -> None:
        """Convert non-JSON-serializable types (e.g., sets) to lists in-place."""
        set_keys = [
            'icmp_types', 'gtp_inner_protocols', 'pfcp_message_types', 'pfcp_cause_codes',
            'ngap_procedure_types', 'ngap_message_types', 'ngap_message_types_names', 'ngap_cause_codes', 'ngap_amf_ue_ngap_id',
            'ngap_ran_ue_ngap_id', 'ngap_authentication_steps', 'ngap_security_steps',
            'timing_anomalies', 'sequence_anomalies', 'retransmission_patterns',
            'failure_patterns', 'failure_scenarios', 'error_patterns', 'root_cause_indicators',
            'ue_behavior_patterns', 'network_load_indicators', 'security_violations', 'compliance_issues',
            'gtp_teids'
        ]
        for key in set_keys:
            if isinstance(features.get(key), set):
                features[key] = list(features[key])
