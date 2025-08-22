# --- IMPORTANT: Fix for sqlite3 on Linux ---
# This line is crucial for enabling the newer pysqlite3 library
# which is required by ChromaDB.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import getpass
from scapy.all import rdpcap, IP, TCP, UDP, DNS, Raw
from scapy.all import load_contrib
import struct
from typing import List, Dict, Tuple, Optional

# --- Corrected Imports from the new LangChain ecosystem ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- ENHANCEMENT: Load Scapy's contributed protocols dynamically and import correctly ---
# We load the contrib modules and then import the layers directly from there.
# Initialize protocol support flags
PFCP_AVAILABLE = False
GTP_AVAILABLE = False
S1AP_AVAILABLE = False
NGAP_AVAILABLE = False

try:
    load_contrib("pfcp")
    from scapy.contrib.pfcp import PFCP
    PFCP_AVAILABLE = True
    print("✓ PFCP protocol support loaded successfully")
except ImportError as e:
    print(f"⚠ PFCP protocol not available: {e}")
    PFCP = None

try:
    load_contrib("gtp")
    from scapy.contrib.gtp import GTP_U_Header
    GTP_AVAILABLE = True
    print("✓ GTP-U protocol support loaded successfully")
except ImportError as e:
    print(f"⚠ GTP-U protocol not available: {e}")
    GTP_U_Header = None

try:
    load_contrib("s1ap")
    from scapy.contrib.s1ap import S1AP
    S1AP_AVAILABLE = True
    print("✓ S1AP protocol support loaded successfully")
except ImportError as e:
    print(f"⚠ S1AP protocol not available: {e}")
    S1AP = None

try:
    load_contrib("ngap")
    from scapy.contrib.ngap import NGAP
    NGAP_AVAILABLE = True
    print("✓ NGAP protocol support loaded successfully")
except ImportError as e:
    print(f"⚠ NGAP protocol not available: {e}")
    NGAP = None

# Check if at least PFCP is available (core requirement)
if not PFCP_AVAILABLE:
    print("❌ PFCP protocol is required but not available. Exiting.")
    sys.exit(1)

print(f"Protocol support: PFCP={PFCP_AVAILABLE}, GTP-U={GTP_AVAILABLE}, S1AP={S1AP_AVAILABLE}, NGAP={NGAP_AVAILABLE}")

class PFCPAnalyzer:
    """
    Enhanced PFCP protocol analyzer for in-depth telecom protocol analysis
    """
    
    # PFCP Message Types with detailed descriptions
    # Based on 3GPP TS 29.244 - Official PFCP Protocol Specification
    # Source: Scapy contrib PFCP module and 3GPP standards
    PFCP_MESSAGE_TYPES = {
        # Node Management Messages
        1: "Heartbeat Request",
        2: "Heartbeat Response", 
        3: "PFD Management Request",
        4: "PFD Management Response",
        5: "Association Setup Request",
        6: "Association Setup Response",
        7: "Association Update Request",
        8: "Association Update Response",
        9: "Association Release Request",
        10: "Association Release Response",
        11: "Version Not Supported Response",
        12: "Node Report Request",
        13: "Node Report Response",
        14: "Session Set Deletion Request",
        15: "Session Set Deletion Response",
        
        # Session Management Messages (Official 3GPP TS 29.244)
        50: "Session Establishment Request",
        51: "Session Establishment Response",
        52: "Session Modification Request",
        53: "Session Modification Response",
        54: "Session Deletion Request",
        55: "Session Deletion Response",
        56: "Session Report Request",
        57: "Session Report Response",
        
        # Reserved ranges for future use
        16: "Reserved",
        17: "Reserved",
        18: "Reserved",
        19: "Reserved",
        20: "Reserved",
        21: "Reserved",
        22: "Reserved",
        23: "Reserved",
        24: "Reserved",
        25: "Reserved",
        26: "Reserved",
        27: "Reserved",
        28: "Reserved",
        29: "Reserved",
        30: "Reserved",
        31: "Reserved",
        32: "Reserved",
        33: "Reserved",
        34: "Reserved",
        35: "Reserved",
        36: "Reserved",
        37: "Reserved",
        38: "Reserved",
        39: "Reserved",
        40: "Reserved",
        41: "Reserved",
        42: "Reserved",
        43: "Reserved",
        44: "Reserved",
        45: "Reserved",
        46: "Reserved",
        47: "Reserved",
        48: "Reserved",
        49: "Reserved",
        
        # Additional 3GPP Release messages (if any)
        58: "Reserved",
        59: "Reserved",
        60: "Reserved",
        61: "Reserved",
        62: "Reserved",
        63: "Reserved",
        64: "Reserved",
        65: "Reserved",
        66: "Reserved",
        67: "Reserved",
        68: "Reserved",
        69: "Reserved",
        70: "Reserved",
        71: "Reserved",
        72: "Reserved",
        73: "Reserved",
        74: "Reserved",
        75: "Reserved",
        76: "Reserved",
        77: "Reserved",
        78: "Reserved",
        79: "Reserved",
        80: "Reserved",
        81: "Reserved",
        82: "Reserved",
        83: "Reserved",
        84: "Reserved",
        85: "Reserved",
        86: "Reserved",
        87: "Reserved",
        88: "Reserved",
        89: "Reserved",
        90: "Reserved",
        91: "Reserved",
        92: "Reserved",
        93: "Reserved",
        94: "Reserved",
        95: "Reserved",
        96: "Reserved",
        97: "Reserved",
        98: "Reserved",
        99: "Reserved",
        100: "Reserved"
    }
    
    # PFCP Information Elements
    PFCP_IE_TYPES = {
        1: "Create PDR",
        2: "PDI",
        3: "Create FAR",
        4: "Forwarding Parameters",
        5: "Duplicating Parameters",
        6: "Create URR",
        7: "Create QER",
        8: "Created PDR",
        9: "Update PDR",
        10: "Update FAR",
        11: "Update Forwarding Parameters",
        12: "Update Duplicating Parameters",
        13: "Update URR",
        14: "Update QER",
        15: "Remove PDR",
        16: "Remove FAR",
        17: "Remove URR",
        18: "Remove QER",
        19: "Cause",
        20: "Source Interface",
        21: "F-TEID",
        22: "Network Instance",
        23: "SDF Filter",
        24: "Application ID",
        25: "Gate Status",
        26: "MBR",
        27: "GBR",
        28: "QER Correlation ID",
        29: "Precedence",
        30: "Transport Level Marking",
        31: "Volume Threshold",
        32: "Time Threshold",
        33: "Monitoring Time",
        34: "Subsequent Volume Threshold",
        35: "Subsequent Time Threshold",
        36: "Inactivity Detection Time",
        37: "Reporting Triggers",
        38: "Redirect Information",
        39: "Report Type",
        40: "Offending IE",
        41: "Forwarding Policy",
        42: "Destination Interface",
        43: "UP Function Features",
        44: "Apply Action",
        45: "Downlink Data Service Information",
        46: "Downlink Data Notification Delay",
        47: "DL Buffering Duration",
        48: "DL Buffering Suggested Packet Count",
        49: "PFCPSMReq-Flags",
        50: "PFCPSRRsp-Flags",
        51: "Sequence Number",
        52: "Metric",
        53: "Timer",
        54: "PDR ID",
        55: "F-SEID",
        56: "Application Instance ID",
        57: "Flow Information",
        58: "UE IP Address",
        59: "Packet Rate",
        60: "Outer Header Creation",
        61: "BAR ID",
        62: "CP Function Features",
        63: "Usage Information",
        64: "Application Instance ID",
        65: "Flow Information",
        66: "UE IP Address",
        67: "Packet Rate",
        68: "Outer Header Creation",
        69: "BAR ID",
        70: "CP Function Features",
        71: "Usage Information",
        72: "Application Instance ID",
        73: "Flow Information",
        74: "UE IP Address",
        75: "Packet Rate",
        76: "Outer Header Creation",
        77: "BAR ID",
        78: "CP Function Features",
        79: "Usage Information",
        80: "Application Instance ID"
    }
    
    def __init__(self):
        self.session_establishment_requests = []
        self.session_establishment_responses = []
        self.session_modification_requests = []
        self.session_modification_responses = []
        self.session_deletion_requests = []
        self.session_deletion_responses = []
        self.association_requests = []
        self.association_responses = []
        self.heartbeat_messages = []
        
    def analyze_pfcp_packet(self, packet_data: bytes, packet_info: Dict) -> Dict:
        """
        Comprehensive PFCP packet analysis
        """
        analysis = {
            "message_type": None,
            "message_name": None,
            "seid": None,
            "sequence_number": None,
            "information_elements": [],
            "is_request": False,
            "is_response": False,
            "is_successful": None,
            "cause_code": None,
            "raw_data": packet_data.hex(),
            "packet_info": packet_info
        }
        
        if len(packet_data) < 8:  # Minimum PFCP header size
            analysis["error"] = "Packet too short for PFCP header"
            return analysis
            
        try:
            # Parse PFCP header (first 8 bytes)
            # Version (3 bits), Spare (1 bit), Message Type (8 bits), Length (16 bits), SEID (64 bits)
            version = (packet_data[0] >> 5) & 0x07
            message_type = packet_data[1]
            length = struct.unpack('!H', packet_data[2:4])[0]
            seid = struct.unpack('!Q', packet_data[4:12])[0] if len(packet_data) >= 12 else None
            
            analysis["message_type"] = message_type
            analysis["message_name"] = self.PFCP_MESSAGE_TYPES.get(message_type, f"Unknown Type {message_type}")
            analysis["seid"] = seid
            analysis["length"] = length
            
            # Determine if it's request or response
            # Request messages: odd numbers (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89)
            # Response messages: even numbers (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90)
            if message_type % 2 == 1:  # Odd numbers are requests
                analysis["is_request"] = True
                analysis["is_response"] = False
            else:  # Even numbers are responses
                analysis["is_request"] = False
                analysis["is_response"] = True
                
                # For response messages, check if they were successful
                if message_type == 6:  # Association Setup Response
                    analysis["is_successful"] = self._check_association_success(packet_data)
                elif message_type == 51:  # Session Establishment Response
                    analysis["is_successful"] = self._check_session_establishment_success(packet_data)
                elif message_type == 53:  # Session Modification Response
                    analysis["is_successful"] = self._check_session_modification_success(packet_data)
                elif message_type == 55:  # Session Deletion Response
                    analysis["is_successful"] = self._check_session_deletion_success(packet_data)
                elif message_type in [2, 4, 8, 10, 12, 14, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90]:  # Other response messages
                    analysis["is_successful"] = self._check_generic_response_success(packet_data)
            
            # Parse Information Elements if packet is long enough
            if len(packet_data) > 12:
                analysis["information_elements"] = self._parse_information_elements(packet_data[12:])
            
            # Store in appropriate category
            if message_type == 50:  # Session Establishment Request
                self.session_establishment_requests.append(analysis)
            elif message_type == 51:  # Session Establishment Response
                self.session_establishment_responses.append(analysis)
            elif message_type == 52:  # Session Modification Request
                self.session_modification_requests.append(analysis)
            elif message_type == 53:  # Session Modification Response
                self.session_modification_responses.append(analysis)
            elif message_type == 54:  # Session Deletion Request
                self.session_deletion_requests.append(analysis)
            elif message_type == 55:  # Session Deletion Response
                self.session_deletion_responses.append(analysis)
            elif message_type == 5:  # Association Setup Request
                self.association_requests.append(analysis)
            elif message_type == 6:  # Association Setup Response
                self.association_responses.append(analysis)
            elif message_type in [1, 2]:  # Heartbeat messages
                self.heartbeat_messages.append(analysis)
                
        except Exception as e:
            analysis["error"] = f"Error parsing PFCP packet: {e}"
            
        return analysis
    
    def _check_association_success(self, packet_data: bytes) -> bool:
        """Check if association setup was successful by looking for Cause IE"""
        try:
            if len(packet_data) < 12:
                return False
            
            # Look for Cause IE (type 19) in the packet
            # PFCP Cause IE format: Type(2 bytes) + Length(2 bytes) + Cause Value(1 byte) + Spare(1 byte)
            offset = 12  # Start after PFCP header
            
            while offset < len(packet_data) - 3:
                if offset + 4 > len(packet_data):
                    break
                    
                ie_type = struct.unpack('!H', packet_data[offset:offset+2])[0]
                ie_length = struct.unpack('!H', packet_data[offset+2:offset+4])[0]
                
                if ie_type == 19:  # Cause IE
                    if offset + 4 + ie_length <= len(packet_data):
                        cause_value = packet_data[offset+4]
                        # Cause value 1 = Request accepted, 2 = Request accepted partially
                        # Other values indicate various failure reasons
                        return cause_value in [1, 2]
                    break
                    
                offset += 4 + ie_length
            
            # If no Cause IE found, assume success (common in successful responses)
            return True
            
        except Exception as e:
            print(f"Error checking association success: {e}")
            return False
    
    def _check_session_establishment_success(self, packet_data: bytes) -> bool:
        """Check if session establishment was successful by looking for Cause IE"""
        try:
            if len(packet_data) < 12:
                return False
            
            # Look for Cause IE (type 19) in the packet
            offset = 12  # Start after PFCP header
            
            while offset < len(packet_data) - 3:
                if offset + 4 > len(packet_data):
                    break
                    
                ie_type = struct.unpack('!H', packet_data[offset:offset+2])[0]
                ie_length = struct.unpack('!H', packet_data[offset+2:offset+4])[0]
                
                if ie_type == 19:  # Cause IE
                    if offset + 4 + ie_length <= len(packet_data):
                        cause_value = packet_data[offset+4]
                        # Cause value 1 = Request accepted, 2 = Request accepted partially
                        # Other values indicate various failure reasons
                        return cause_value in [1, 2]
                    break
                    
                offset += 4 + ie_length
            
            # If no Cause IE found, assume success (common in successful responses)
            return True
            
        except Exception as e:
            print(f"Error checking session establishment success: {e}")
            return False
    
    def _check_session_modification_success(self, packet_data: bytes) -> bool:
        """Check if session modification was successful by looking for Cause IE"""
        try:
            if len(packet_data) < 12:
                return False
            
            # Look for Cause IE (type 19) in the packet
            offset = 12  # Start after PFCP header
            
            while offset < len(packet_data) - 3:
                if offset + 4 > len(packet_data):
                    break
                    
                ie_type = struct.unpack('!H', packet_data[offset:offset+2])[0]
                ie_length = struct.unpack('!H', packet_data[offset+2:offset+4])[0]
                
                if ie_type == 19:  # Cause IE
                    if offset + 4 + ie_length <= len(packet_data):
                        cause_value = packet_data[offset+4]
                        return cause_value in [1, 2]
                    break
                    
                offset += 4 + ie_length
            
            return True
            
        except Exception as e:
            print(f"Error checking session modification success: {e}")
            return False
    
    def _check_session_deletion_success(self, packet_data: bytes) -> bool:
        """Check if session deletion was successful by looking for Cause IE"""
        try:
            if len(packet_data) < 12:
                return False
            
            # Look for Cause IE (type 19) in the packet
            offset = 12  # Start after PFCP header
            
            while offset < len(packet_data) - 3:
                if offset + 4 > len(packet_data):
                    break
                    
                ie_type = struct.unpack('!H', packet_data[offset:offset+2])[0]
                ie_length = struct.unpack('!H', packet_data[offset+2:offset+4])[0]
                
                if ie_type == 19:  # Cause IE
                    if offset + 4 + ie_length <= len(packet_data):
                        cause_value = packet_data[offset+4]
                        return cause_value in [1, 2]
                    break
                    
                offset += 4 + ie_length
            
            return True
            
        except Exception as e:
            print(f"Error checking session deletion success: {e}")
            return False
    
    def _check_generic_response_success(self, packet_data: bytes) -> bool:
        """Check if a generic PFCP response was successful by looking for Cause IE"""
        try:
            if len(packet_data) < 12:
                return False
            
            # Look for Cause IE (type 19) in the packet
            offset = 12  # Start after PFCP header
            
            while offset < len(packet_data) - 3:
                if offset + 4 > len(packet_data):
                    break
                    
                ie_type = struct.unpack('!H', packet_data[offset:offset+2])[0]
                ie_length = struct.unpack('!H', packet_data[offset+2:offset+4])[0]
                
                if ie_type == 19:  # Cause IE
                    if offset + 4 + ie_length <= len(packet_data):
                        cause_value = packet_data[offset+4]
                        return cause_value in [1, 2]
                    break
                    
                offset += 4 + ie_length
            
            return True
            
        except Exception as e:
            print(f"Error checking generic response success: {e}")
            return False
    
    def _parse_information_elements(self, ie_data: bytes) -> List[Dict]:
        """Parse PFCP Information Elements"""
        ies = []
        offset = 0
        
        while offset < len(ie_data):
            if offset + 4 > len(ie_data):
                break
                
            try:
                ie_type = struct.unpack('!H', ie_data[offset:offset+2])[0]
                ie_length = struct.unpack('!H', ie_data[offset+2:offset+4])[0]
                
                ie_info = {
                    "type": ie_type,
                    "type_name": self.PFCP_IE_TYPES.get(ie_type, f"Unknown IE {ie_type}"),
                    "length": ie_length,
                    "data": ie_data[offset+4:offset+4+ie_length].hex() if offset+4+ie_length <= len(ie_data) else ""
                }
                
                ies.append(ie_info)
                offset += 4 + ie_length
                
            except Exception as e:
                ies.append({"error": f"Error parsing IE at offset {offset}: {e}"})
                break
                
        return ies
    
    def get_session_establishment_analysis(self) -> Dict:
        """Analyze session establishment success/failure patterns"""
        analysis = {
            "total_requests": len(self.session_establishment_requests),
            "total_responses": len(self.session_establishment_responses),
            "successful_sessions": 0,
            "failed_sessions": 0,
            "session_pairs": [],
            "analysis_summary": "",
            "detailed_analysis": []
        }
        
        # Match requests with responses based on SEID and timing
        for request in self.session_establishment_requests:
            matching_response = None
            for response in self.session_establishment_responses:
                if response["seid"] == request["seid"]:
                    matching_response = response
                    break
            
            if matching_response:
                session_pair = {
                    "request": request,
                    "response": matching_response,
                    "successful": matching_response.get("is_successful", False)
                }
                analysis["session_pairs"].append(session_pair)
                
                # Detailed analysis for each session pair
                detailed_info = {
                    "request_packet": request["packet_info"].get("packet_number", "Unknown"),
                    "request_seid": request["seid"],
                    "request_timestamp": request["packet_info"].get("timestamp", "Unknown"),
                    "response_packet": matching_response["packet_info"].get("packet_number", "Unknown"),
                    "response_seid": matching_response["seid"],
                    "response_timestamp": matching_response["packet_info"].get("timestamp", "Unknown"),
                    "success": matching_response.get("is_successful", False),
                    "cause_code": self._extract_cause_code(matching_response["raw_data"])
                }
                analysis["detailed_analysis"].append(detailed_info)
                
                if session_pair["successful"]:
                    analysis["successful_sessions"] += 1
                else:
                    analysis["failed_sessions"] += 1
        
        # Generate analysis summary
        if analysis["total_requests"] > 0:
            success_rate = (analysis["successful_sessions"] / analysis["total_requests"]) * 100
            analysis["analysis_summary"] = f"Session establishment success rate: {success_rate:.1f}% ({analysis['successful_sessions']}/{analysis['total_requests']})"
        else:
            analysis["analysis_summary"] = "No session establishment requests found"
            
        return analysis
    
    def _extract_cause_code(self, raw_data_hex: str) -> str:
        """Extract and interpret PFCP Cause code from raw data"""
        try:
            raw_data = bytes.fromhex(raw_data_hex)
            if len(raw_data) < 16:  # Need at least PFCP header + some IEs
                return "Unknown - insufficient data"
            
            offset = 12  # Start after PFCP header
            while offset < len(raw_data) - 3:
                if offset + 4 > len(raw_data):
                    break
                    
                ie_type = struct.unpack('!H', raw_data[offset:offset+2])[0]
                ie_length = struct.unpack('!H', raw_data[offset+2:offset+4])[0]
                
                if ie_type == 19:  # Cause IE
                    if offset + 4 + ie_length <= len(raw_data):
                        cause_value = raw_data[offset+4]
                        cause_meaning = self._get_cause_meaning(cause_value)
                        return f"{cause_value} - {cause_meaning}"
                    break
                    
                offset += 4 + ie_length
            
            return "No Cause IE found - assumed success"
            
        except Exception as e:
            return f"Error extracting cause: {e}"
    
    def _get_cause_meaning(self, cause_value: int) -> str:
        """Get human-readable meaning of PFCP Cause codes"""
        cause_meanings = {
            1: "Request accepted",
            2: "Request accepted partially",
            3: "Request rejected",
            4: "Session context not found",
            5: "Mandatory IE missing",
            6: "Conditional IE missing",
            7: "Unsupported message",
            8: "Message format error",
            9: "Unsupported IE",
            10: "IE value error",
            11: "Mandatory IE missing",
            12: "Conditional IE missing",
            13: "Unsupported message",
            14: "Message format error",
            15: "Unsupported IE",
            16: "IE value error",
            17: "Unsupported message",
            18: "Message format error",
            19: "Unsupported IE",
            20: "IE value error"
        }
        return cause_meanings.get(cause_value, f"Unknown cause code {cause_value}")
    
    def get_comprehensive_pfcp_analysis(self) -> Dict:
        """Get comprehensive PFCP protocol analysis"""
        return {
            "session_establishment": self.get_session_establishment_analysis(),
            "session_modification": {
                "total_requests": len(self.session_modification_requests),
                "total_responses": len(self.session_modification_responses),
                "successful_modifications": sum(1 for resp in self.session_modification_responses if resp.get("is_successful", False)),
                "failed_modifications": sum(1 for resp in self.session_modification_responses if not resp.get("is_successful", True))
            },
            "session_deletion": {
                "total_requests": len(self.session_deletion_requests),
                "total_responses": len(self.session_deletion_responses),
                "successful_deletions": sum(1 for resp in self.session_deletion_responses if resp.get("is_successful", False)),
                "failed_deletions": sum(1 for resp in self.session_deletion_responses if not resp.get("is_successful", True))
            },
            "association_analysis": {
                "total_requests": len(self.association_requests),
                "total_responses": len(self.association_responses),
                "successful_associations": sum(1 for resp in self.association_responses if resp.get("is_successful", False)),
                "failed_associations": sum(1 for resp in self.association_responses if not resp.get("is_successful", True))
            },
            "heartbeat_analysis": {
                "total_heartbeats": len(self.heartbeat_messages),
                "requests": len([h for h in self.heartbeat_messages if h.get("is_request", False)]),
                "responses": len([h for h in self.heartbeat_messages if h.get("is_response", False)])
            },
            "total_pfcp_packets": (len(self.session_establishment_requests) + len(self.session_establishment_responses) + 
                                  len(self.session_modification_requests) + len(self.session_modification_responses) +
                                  len(self.session_deletion_requests) + len(self.session_deletion_responses) +
                                  len(self.association_requests) + len(self.association_responses) + 
                                  len(self.heartbeat_messages))
        }

def parse_pcap_to_text(pcap_file: str) -> List[str]:
    """
    Enhanced PCAP parser with comprehensive telecom protocol analysis
    """
    try:
        packets = rdpcap(pcap_file)
    except FileNotFoundError:
        print(f"Error: The file '{pcap_file}' was not found.")
        return []
    except Exception as e:
        print(f"Error reading pcap file: {e}")
        return []

    documents = []
    total_packets = len(packets)
    ip_packets = 0
    pfcp_packets = 0
    udp_packets = 0
    tcp_packets = 0
    
    # Initialize PFCP analyzer
    pfcp_analyzer = PFCPAnalyzer()
    
    print(f"Total packets in PCAP: {total_packets}")
    
    for i, packet in enumerate(packets):
        text_lines = []
        packet_analysis = {}
        
        # Extract timestamp
        text_lines.append(f"Packet #{i+1} - Timestamp: {packet.time}")
        
        # Check for IP layer
        if packet.haslayer(IP):
            ip_packets += 1
            ip_layer = packet[IP]
            text_lines.append(f"  Source IP: {ip_layer.src}")
            text_lines.append(f"  Destination IP: {ip_layer.dst}")
            text_lines.append(f"  TTL: {ip_layer.ttl}")
            text_lines.append(f"  Protocol: {ip_layer.proto}")

            # Check for TCP or UDP layers
            if packet.haslayer(TCP):
                tcp_packets += 1
                tcp_layer = packet[TCP]
                text_lines.append(f"  Transport: TCP")
                text_lines.append(f"  Source Port: {tcp_layer.sport}")
                text_lines.append(f"  Destination Port: {tcp_layer.dport}")
                text_lines.append(f"  Flags: {tcp_layer.flags}")
                text_lines.append(f"  Sequence: {tcp_layer.seq}")
                text_lines.append(f"  Acknowledgment: {tcp_layer.ack}")
                text_lines.append(f"  Window: {tcp_layer.window}")
                text_lines.append(f"  Payload length: {len(tcp_layer.payload)}")
                
                # Check for specific TCP protocols
                if tcp_layer.dport == 8080 or tcp_layer.sport == 8080:
                    text_lines.append(f"  Application: HTTP/HTTPS")
                elif tcp_layer.dport == 22 or tcp_layer.sport == 22:
                    text_lines.append(f"  Application: SSH")
                elif tcp_layer.dport == 53 or tcp_layer.sport == 53:
                    text_lines.append(f"  Application: DNS")
                    
            elif packet.haslayer(UDP):
                udp_packets += 1
                udp_layer = packet[UDP]
                text_lines.append(f"  Transport: UDP")
                text_lines.append(f"  Source Port: {udp_layer.sport}")
                text_lines.append(f"  Destination Port: {udp_layer.dport}")
                text_lines.append(f"  Payload length: {len(udp_layer.payload)}")
                
                # --- ENHANCED: Comprehensive PFCP Analysis ---
                if PFCP_AVAILABLE and packet.haslayer(PFCP):
                    pfcp_packets += 1
                    pfcp_layer = packet[PFCP]
                    text_lines.append(f"  5G Protocol: PFCP (Packet Forwarding Control Protocol)")
                    
                    # Get raw PFCP data for detailed analysis
                    raw_pfcp_data = bytes(pfcp_layer)
                    if hasattr(pfcp_layer, 'raw') and pfcp_layer.raw:
                        raw_pfcp_data = bytes(pfcp_layer.raw)
                    
                    # Analyze PFCP packet in detail
                    pfcp_analysis = pfcp_analyzer.analyze_pfcp_packet(raw_pfcp_data, {
                        "packet_number": i+1,
                        "timestamp": packet.time,
                        "source_ip": ip_layer.src,
                        "destination_ip": ip_layer.dst,
                        "source_port": udp_layer.sport,
                        "destination_port": udp_layer.dport
                    })
                    
                    # Add detailed PFCP information
                    text_lines.append(f"  PFCP Message Type: {pfcp_analysis['message_type']} - {pfcp_analysis['message_name']}")
                    text_lines.append(f"  PFCP SEID: {pfcp_analysis['seid']}")
                    text_lines.append(f"  PFCP Length: {pfcp_analysis.get('length', 'Unknown')}")
                    text_lines.append(f"  PFCP Direction: {'Request' if pfcp_analysis['is_request'] else 'Response' if pfcp_analysis['is_response'] else 'Unknown'}")
                    
                    if pfcp_analysis['is_response'] and pfcp_analysis['is_successful'] is not None:
                        text_lines.append(f"  PFCP Success: {'Yes' if pfcp_analysis['is_successful'] else 'No'}")
                    
                    # Add Information Elements if available
                    if pfcp_analysis['information_elements']:
                        text_lines.append(f"  PFCP Information Elements: {len(pfcp_analysis['information_elements'])}")
                        for ie in pfcp_analysis['information_elements'][:5]:  # Show first 5 IEs
                            text_lines.append(f"    IE {ie['type']}: {ie['type_name']} ({ie['length']} bytes)")
                    
                    # Add raw data for debugging
                    text_lines.append(f"  PFCP Raw Data: {raw_pfcp_data[:100].hex()}")
                    
                elif GTP_AVAILABLE and packet.haslayer(GTP_U_Header):
                    gtpu_layer = packet[GTP_U_Header]
                    text_lines.append(f"  5G Protocol: GTP-U (GPRS Tunneling Protocol - User Plane)")
                    try:
                        if hasattr(gtpu_layer, 'teid'):
                            text_lines.append(f"  GTP-U TEID: {gtpu_layer.teid}")
                        else:
                            text_lines.append(f"  GTP-U TEID: Not accessible")
                        text_lines.append(f"  GTP-U Version: {gtpu_layer.version}")
                        text_lines.append(f"  GTP-U Message Type: {gtpu_layer.gtp_type}")
                    except Exception as e:
                        text_lines.append(f"  GTP-U Error: {e}")
                        
                # Check for other 5G protocols
                elif S1AP_AVAILABLE and packet.haslayer(S1AP):
                    text_lines.append(f"  5G Protocol: S1AP (S1 Application Protocol)")
                elif NGAP_AVAILABLE and packet.haslayer(NGAP):
                    text_lines.append(f"  5G Protocol: NGAP (NG Application Protocol)")
                    
                # Check for DNS within UDP
                elif packet.haslayer(DNS):
                    dns_layer = packet[DNS]
                    text_lines.append(f"  Application: DNS")
                    if dns_layer.qd:
                        try:
                            query_name = dns_layer.qd.qname.decode()
                            text_lines.append(f"  DNS Query: {query_name}")
                        except:
                            text_lines.append(f"  DNS Query: Unable to decode")
                    else:
                        text_lines.append(f"  DNS Query: No query")
                    text_lines.append(f"  DNS Type: {dns_layer.qd.qtype if dns_layer.qd else 'Unknown'}")
                    
            # Add packet size information
            text_lines.append(f"  Total Packet Size: {len(packet)} bytes")
            
            # Check for Raw layer for additional data
            if packet.haslayer(Raw):
                raw_layer = packet[Raw]
                raw_data = bytes(raw_layer)
                if len(raw_data) > 0:
                    text_lines.append(f"  Raw Payload: {raw_data[:50].hex()}")
                    text_lines.append(f"  Raw Payload Length: {len(raw_data)} bytes")
            
        else:
            # Handle non-IP packets
            text_lines.append(f"  No IP layer detected")
            text_lines.append(f"  Packet Type: {packet.__class__.__name__}")
            text_lines.append(f"  Packet Size: {len(packet)} bytes")
            
            # Check if it's a raw PFCP packet
            if PFCP_AVAILABLE and packet.haslayer(PFCP):
                pfcp_packets += 1
                text_lines.append(f"  Raw PFCP packet detected")
                pfcp_layer = packet[PFCP]
                text_lines.append(f"  5G Protocol: PFCP (Control Plane)")
                
                # Analyze raw PFCP packet
                raw_pfcp_data = bytes(pfcp_layer)
                if hasattr(pfcp_layer, 'raw') and pfcp_layer.raw:
                    raw_pfcp_data = bytes(pfcp_layer.raw)
                
                pfcp_analysis = pfcp_analyzer.analyze_pfcp_packet(raw_pfcp_data, {
                    "packet_number": i+1,
                    "timestamp": packet.time,
                    "raw_packet": True
                })
                
                text_lines.append(f"  PFCP Message Type: {pfcp_analysis['message_type']} - {pfcp_analysis['message_name']}")
                text_lines.append(f"  PFCP Raw Length: {len(raw_pfcp_data)} bytes")
                text_lines.append(f"  PFCP Raw Data: {raw_pfcp_data[:100].hex()}")
        
        documents.append("\n".join(text_lines))
    
    print(f"Packets with IP layer: {ip_packets}")
    print(f"Packets with TCP layer: {tcp_packets}")
    print(f"Packets with UDP layer: {udp_packets}")
    print(f"Packets with PFCP layer: {pfcp_packets}")
    print(f"Total documents created: {len(documents)}")
    
    # Print PFCP analysis summary
    if pfcp_packets > 0:
        print("\n=== PFCP Analysis Summary ===")
        comprehensive_analysis = pfcp_analyzer.get_comprehensive_pfcp_analysis()
        session_analysis = comprehensive_analysis['session_establishment']
        
        print(f"Total PFCP Packets: {comprehensive_analysis['total_pfcp_packets']}")
        print(f"Session Establishment Requests: {session_analysis['total_requests']}")
        print(f"Session Establishment Responses: {session_analysis['total_responses']}")
        print(f"Successful Sessions: {session_analysis['successful_sessions']}")
        print(f"Failed Sessions: {session_analysis['failed_sessions']}")
        print(f"Association Requests: {comprehensive_analysis['association_analysis']['total_requests']}")
        print(f"Association Responses: {comprehensive_analysis['association_analysis']['total_responses']}")
        print(f"Successful Associations: {comprehensive_analysis['association_analysis']['successful_associations']}")
        print(f"Failed Associations: {comprehensive_analysis['association_analysis']['failed_associations']}")
        print(f"Heartbeat Messages: {comprehensive_analysis['heartbeat_analysis']['total_heartbeats']}")
        print(f"Analysis: {session_analysis['analysis_summary']}")
        
        # Add comprehensive PFCP analysis to documents
        pfcp_summary_doc = f"""
PFCP Protocol Analysis Summary:
- Total PFCP Packets: {comprehensive_analysis['total_pfcp_packets']}
- Session Establishment Requests: {session_analysis['total_requests']}
- Session Establishment Responses: {session_analysis['total_responses']}
- Successful Sessions: {session_analysis['successful_sessions']}
- Failed Sessions: {session_analysis['failed_sessions']}
- Success Rate: {(session_analysis['successful_sessions'] / max(session_analysis['total_requests'], 1)) * 100:.1f}%
- Association Requests: {comprehensive_analysis['association_analysis']['total_requests']}
- Association Responses: {comprehensive_analysis['association_analysis']['total_responses']}
- Successful Associations: {comprehensive_analysis['association_analysis']['successful_associations']}
- Failed Associations: {comprehensive_analysis['association_analysis']['failed_associations']}
- Heartbeat Messages: {comprehensive_analysis['heartbeat_analysis']['total_heartbeats']}

Detailed Session Analysis:
"""
        for i, detailed_info in enumerate(session_analysis['detailed_analysis']):
            pfcp_summary_doc += f"""
Session Pair {i+1}:
- Request: Packet #{detailed_info['request_packet']} - Session Establishment Request (SEID: {detailed_info['request_seid']})
- Response: Packet #{detailed_info['response_packet']} - Session Establishment Response (SEID: {detailed_info['response_seid']})
- Success: {'Yes' if detailed_info['success'] else 'No'}
- Cause Code: {detailed_info['cause_code']}
- Request Timestamp: {detailed_info['request_timestamp']}
- Response Timestamp: {detailed_info['response_timestamp']}
"""
        
        documents.append(pfcp_summary_doc)
    
    return documents

def main():
    """
    Main function to run the RAG application with a chat interface.
    """
    try:
        # --- 2. Data Ingestion from pcap file ---
        pcap_path = input("Please enter the path to your .pcap file: ").strip()
        if not pcap_path:
            print("No file path entered. Exiting.")
            return

        print(f"Processing pcap file: {pcap_path}")
        documents = parse_pcap_to_text(pcap_path)
        if not documents:
            print("No documents were parsed. Exiting.")
            return

        print(f"Successfully parsed {len(documents)} packets.")
        
        # Show first few packets for verification
        print("\nFirst 3 packets preview:")
        for i, doc in enumerate(documents[:3]):
            print(f"\n--- Packet {i+1} ---")
            print(doc[:500] + "..." if len(doc) > 500 else doc)
        
        # Show PFCP packets specifically
        pfcp_docs = [doc for doc in documents if "PFCP" in doc]
        print(f"\nFound {len(pfcp_docs)} PFCP packets in documents")
        if pfcp_docs:
            print("\nFirst PFCP packet preview:")
            print(pfcp_docs[0][:500] + "..." if len(pfcp_docs[0]) > 500 else pfcp_docs[0])

        # --- 3. Indexing (Embedding and Storing) ---
        print("\nCreating vector store...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        # Create documents from the text strings
        from langchain_core.documents import Document
        docs = [Document(page_content=doc, metadata={}) for doc in documents]
        texts = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings
        )
        print(f"Indexed {len(texts)} chunks of network data.")
        
        # Debug: Check what's in the vector store
        print(f"\nVector store contains {len(texts)} text chunks")
        print("Sample chunks:")
        for i, text in enumerate(texts[:3]):
            print(f"\n--- Chunk {i+1} ---")
            print(text.page_content[:200] + "..." if len(text.page_content) > 200 else text.page_content)

        # --- 4. Retrieval & Generation ---
        print("Setting up LLM...")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        rag_prompt_template = """
        You are a network analysis assistant specializing in 5G PFCP (Packet Forwarding Control Protocol) analysis. Your role is to analyze and summarize network traffic logs provided as context.
        You must answer the question based ONLY on the provided context. If the answer is not in the context, say "I could not find the answer in the provided network logs."
        
        IMPORTANT: When counting packets or analyzing network traffic, make sure to examine ALL provided context thoroughly.
        If you are asked about packet counts, analyze every single packet in the context to provide an accurate count.
        Do not limit your analysis to just a few packets - use all available information.
        
        For PFCP connection analysis:
        - Look for message type patterns (Request/Response pairs)
        - Heartbeat messages (types 1-2) indicate ongoing connection health
        - Association messages (types 3-8) indicate connection establishment/management
        - Session messages (types 14-21) indicate data session handling
        - Check for consistent bidirectional communication between IP addresses
        - Look for error indicators or failed response patterns
        
        SPECIFICALLY for PFCP Session Establishment Analysis:
        - Message Type 50 = Session Establishment Request
        - Message Type 51 = Session Establishment Response
        - Look for matching SEID (Session Endpoint ID) between requests and responses
        - Check if responses indicate success (Cause IE with value 1 or 2) or failure
        - Analyze the sequence of packets to determine if sessions were established successfully
        - Count successful vs failed session establishments
        - Look for any error patterns or failed requests
        
        When asked about PFCP session establishment success:
        1. Count all Session Establishment Request packets (Type 50)
        2. Count all Session Establishment Response packets (Type 51)
        3. Match requests with responses using SEID
        4. Determine success/failure based on Cause IE analysis
        5. Provide detailed statistics and packet-level analysis

        Context:
        {context}

        Question:
        {question}
        """
        rag_prompt = PromptTemplate.from_template(rag_prompt_template)

        # Configure retriever to return more documents
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 50}  # Return up to 50 most relevant documents
        )
        
        print(f"Retriever configured to return up to 50 most relevant documents")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": rag_prompt},
            return_source_documents=True
        )

        # --- 5. Interactive Chat Loop ---
        print("\n=== RAG System Ready! ===")
        print("Ask a question about the network traffic (type 'exit' to quit):")
        while True:
            try:
                user_question = input("> ").strip()
                if user_question.lower() == 'exit':
                    print("Exiting program.")
                    break

                if user_question:
                    print("Thinking...")
                    response = qa_chain.invoke({"query": user_question})
                    print("\nAnswer:")
                    print(response["result"])
                    print("\n" + "="*50 + "\n")

            except Exception as e:
                print(f"An error occurred: {e}")
                break
                
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
