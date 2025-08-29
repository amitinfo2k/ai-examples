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
                'ngap_message_types': [],    # Specific NGAP message type codes
                'ngap_cause_codes': [],     # For reject/error messages
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
                        # NGAP runs over SCTP port 38412
                        try:
                            sctp = pkt[SCTP]
                            if getattr(sctp, 'sport', None) == 38412 or getattr(sctp, 'dport', None) == 38412:
                                # Enhanced NGAP message parsing
                                ngap_info = self._parse_ngap_message(pkt, sctp, pkt_index)
                                features['ngap_messages'].append(ngap_info)
                                
                                # Extract NGAP procedure and message types
                                if ngap_info.get('procedure_code'):
                                    features['ngap_procedure_types'].append(ngap_info['procedure_code'])
                                if ngap_info.get('message_type'):
                                    features['ngap_message_types'].append(ngap_info['message_type'])
                                
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
                                
                                # Track cause codes for failures
                                if ngap_info.get('cause_code'):
                                    features['ngap_cause_codes'].append(ngap_info['cause_code'])
                                    
                        except Exception as e:
                            self.logger.debug(f"NGAP parsing error: {e}")
                            # Fallback to basic info
                            features['ngap_messages'].append({
                                'src_port': getattr(sctp, 'sport', None),
                                'dst_port': getattr(sctp, 'dport', None),
                                'length': len(sctp.payload) if hasattr(sctp, 'payload') else 0
                            })
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

            # Enhanced analysis
            self._detect_failure_patterns(features)
            self._analyze_timing_anomalies(features)
            self._enhance_pfcp_analysis(features)

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
        
        # Enhanced NGAP specific
        if features['ngap_message_count'] > 0:
            desc.append(f"Contains {features['ngap_message_count']} NGAP messages.")
            
            # Add procedure types if available
            if features.get('ngap_procedure_types'):
                procedures = [self._get_procedure_name(proc) for proc in features['ngap_procedure_types']]
                desc.append(f"NGAP procedures: {', '.join(procedures)}")
            
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

    def _parse_ngap_message(self, packet, sctp, packet_index: int) -> Dict:
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
                'is_reject': False
            }
            
            # Parse NGAP payload if available
            if hasattr(sctp, 'payload') and sctp.payload:
                payload_bytes = bytes(sctp.payload)
                
                # NGAP header is typically first 4 bytes
                if len(payload_bytes) >= 4:
                    # First byte: Protocol discriminator (should be 0x00 for NGAP)
                    protocol_discriminator = payload_bytes[0]
                    if protocol_discriminator == 0x00:
                        # Bytes 1-2: Procedure code (big endian)
                        procedure_code = int.from_bytes(payload_bytes[1:3], 'big')
                        ngap_info['procedure_code'] = procedure_code
                        
                        # Byte 3: Message type (big endian)
                        message_type = int.from_bytes(payload_bytes[3:4], 'big')
                        ngap_info['message_type'] = message_type
                        
                        # Map procedure codes to meaningful names
                        # Note: NGAP procedure codes are typically 2 bytes, but some implementations use different ranges
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
                        
                        # Categorize message types
                        if procedure_code in [9, 10]:  # Authentication
                            ngap_info['is_authentication'] = True
                        elif procedure_code in [11, 12, 13]:  # Security
                            ngap_info['is_security'] = True
                        elif procedure_code in [3, 4, 5]:  # Initial Context Setup
                            ngap_info['is_setup'] = True
                        elif procedure_code in [5, 16, 18, 25, 26, 29, 31, 32, 35, 37, 39, 41, 43, 45, 47, 49]:  # Reject/Failure
                            ngap_info['is_reject'] = True
                        
                        # Try to extract cause codes from reject/failure messages
                        if ngap_info['is_reject'] and len(payload_bytes) >= 8:
                            # Look for cause IE (typically after header)
                            for i in range(4, min(len(payload_bytes) - 3, 64)):  # Search first 64 bytes
                                if (i + 3 < len(payload_bytes) and 
                                    payload_bytes[i] == 0x00 and 
                                    payload_bytes[i+1] == 0x15):  # Cause IE type
                                    cause_code = payload_bytes[i+3]
                                    ngap_info['cause_code'] = cause_code
                                    break
                        
                        # Try to extract UE IDs (this is simplified - real parsing would be more complex)
                        # Look for patterns that might indicate UE IDs
                        if len(payload_bytes) >= 12:
                            # This is a simplified approach - real NGAP parsing would be more sophisticated
                            pass
                            
            return ngap_info
            
        except Exception as e:
            self.logger.debug(f"NGAP parsing error in packet {packet_index}: {e}")
            # Return basic info on error
            return {
                'src_port': getattr(sctp, 'sport', None),
                'dst_port': getattr(sctp, 'dport', None),
                'length': len(sctp.payload) if hasattr(sctp, 'payload') else 0
            }

    def _detect_failure_patterns(self, features: Dict) -> None:
        """Detect common failure patterns and categorize them.
        
        Args:
            features: Dictionary of extracted features to update
        """
        failure_patterns = []
        failure_scenarios = []
        error_patterns = []
        root_cause_indicators = []
        
        # NGAP failure detection
        if features.get('ngap_cause_codes'):
            for cause in features['ngap_cause_codes']:
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
        
        # Protocol handshake completion analysis
        if features.get('ngap_procedure_types'):
            # Check if we have both request and response for key procedures
            setup_procedures = [3, 4, 5]  # Initial Context Setup
            auth_procedures = [9, 10]      # Authentication
            security_procedures = [11, 12, 13]  # Security
            
            for proc_type in setup_procedures:
                if proc_type in features['ngap_procedure_types']:
                    if proc_type + 1 in features['ngap_procedure_types'] or proc_type + 2 in features['ngap_procedure_types']:
                        features['protocol_handshake_completion'][f'ngap_setup_{proc_type}'] = 'complete'
                    else:
                        features['protocol_handshake_completion'][f'ngap_setup_{proc_type}'] = 'incomplete'
                        failure_patterns.append(f"NGAP_Setup_Incomplete_Procedure_{proc_type}")
                        failure_scenarios.append("NGAP_Procedure_Incomplete")
                        error_patterns.append("NGAP_Handshake_Incomplete")
                        root_cause_indicators.append("NGAP_Procedure_Timeout_Or_Failure")
        
        # Update features with detected patterns
        features['failure_patterns'] = failure_patterns
        features['failure_scenarios'] = failure_scenarios
        features['error_patterns'] = error_patterns
        features['root_cause_indicators'] = root_cause_indicators
        features['has_failures'] = len(failure_patterns) > 0
        
        # Determine overall registration status
        if features.get('ngap_procedure_types'):
            if features.get('ngap_cause_codes'):
                features['ngap_registration_status'] = 'failed'
            elif any(proc in features['ngap_procedure_types'] for proc in [15, 17]):  # Registration Accept/Complete
                features['ngap_registration_status'] = 'success'
            else:
                features['ngap_registration_status'] = 'partial'

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
        return procedure_names.get(procedure_code, f'UnknownProcedure_{procedure_code}')

    def _normalize_feature_types(self, features: Dict) -> None:
        """Convert non-JSON-serializable types (e.g., sets) to lists in-place."""
        set_keys = [
            'icmp_types', 'gtp_inner_protocols', 'pfcp_message_types', 'pfcp_cause_codes',
            'ngap_procedure_types', 'ngap_message_types', 'ngap_cause_codes', 'ngap_amf_ue_ngap_id',
            'ngap_ran_ue_ngap_id', 'ngap_authentication_steps', 'ngap_security_steps',
            'timing_anomalies', 'sequence_anomalies', 'retransmission_patterns',
            'failure_patterns', 'failure_scenarios', 'error_patterns', 'root_cause_indicators',
            'ue_behavior_patterns', 'network_load_indicators', 'security_violations', 'compliance_issues',
            'gtp_teids'
        ]
        for key in set_keys:
            if isinstance(features.get(key), set):
                features[key] = list(features[key])
