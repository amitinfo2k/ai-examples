"""
Module for generating embeddings from PCAP features and descriptions.
Uses Sentence Transformers for generating embeddings.
"""
from typing import Dict, Any, List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import json

class EmbeddingGenerator:
    """Generates embeddings for PCAP features and descriptions."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self._get_embedding_dim(model_name)
    
    def _get_embedding_dim(self, model_name: str) -> int:
        """Get the embedding dimension for the specified model."""
        # Common embedding dimensions for popular models
        model_dims = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768,
            'paraphrase-multilingual-mpnet-base-v2': 768,
        }
        return model_dims.get(model_name, 768)  # Default to 768 if unknown
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        if not text:
            return [0.0] * self.embedding_dim
            
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_feature_embedding(self, features: Dict[str, Any]) -> List[float]:
        """
        Generate an embedding for PCAP features.
        
        Args:
            features: Dictionary of PCAP features
            
        Returns:
            List of floats representing the feature embedding
        """
        # Convert features to a text representation
        feature_text = self._features_to_text(features)
        return self.generate_embedding(feature_text)
    
    def _features_to_text(self, features: Dict[str, Any]) -> str:
        """
        Convert PCAP features to a text representation for embedding.
        
        Args:
            features: Dictionary of PCAP features
            
        Returns:
            String representation of features
        """
        # Create a structured text representation of the features
        parts = [
            f"Total packets: {features.get('total_packets', 0)}",
            f"Protocol counts: {json.dumps(features.get('protocol_counts', {}))}",
            f"Average packet size: {features.get('avg_packet_size', 0):.2f} bytes",
            f"Packet rate: {features.get('avg_packet_rate', 0):.2f} packets/sec",
        ]
        
        # Add protocol-specific information
        if features.get('gtp_packets', 0) > 0:
            gtp_info = [f"GTP packets: {features['gtp_packets']}"]
            if features.get('gtp_teids'):
                gtp_info.append(f"TEIDs: {len(features['gtp_teids'])} unique")
            if features.get('gtp_inner_protocols'):
                protocol_names = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
                named_protocols = [f"{p}:{protocol_names.get(int(p), 'Unknown')}" for p in sorted(features.get('gtp_inner_protocols', []))]
                gtp_info.append(f"inner protocols: {', '.join(named_protocols)}")
            if features.get('gtp_icmp_packets', 0) > 0:
                gtp_info.append(f"ICMP: {features['gtp_icmp_packets']}")
            if features.get('gtp_non_icmp_packets', 0) > 0:
                gtp_info.append(f"non-ICMP: {features['gtp_non_icmp_packets']}")
            parts.append(" | ".join(gtp_info))
            
        if features.get('pfcp_packets', 0) > 0:
            pfcp_info = [f"PFCP packets: {features['pfcp_packets']}"]
            if features.get('pfcp_message_types'):
                type_list = sorted(features.get('pfcp_message_types', []))
                type_name_map = {
                    50: 'Session Est Req', 51: 'Session Est Resp',
                    52: 'Session Mod Req', 53: 'Session Mod Resp',
                    54: 'Session Del Req', 55: 'Session Del Resp',
                    1: 'Heartbeat Req', 2: 'Heartbeat Resp',
                    5: 'Assoc Setup Req', 6: 'Assoc Setup Resp',
                }
                named = [f"{t}:{type_name_map.get(int(t), 'Unknown')}" for t in type_list]
                pfcp_info.append(f"types: {', '.join(named)}")
            if features.get('pfcp_cause_codes'):
                cause_names = {73: 'Rule creation/modification Failure'}
                named_causes = [f"{c}:{cause_names.get(int(c), 'Unknown')}" for c in sorted(features.get('pfcp_cause_codes', []))]
                pfcp_info.append(f"causes: {', '.join(named_causes)}")
            if features.get('pfcp_session_establishment_failed'):
                pfcp_info.append("session establishment: FAILED")
            if features.get('pfcp_heartbeat_only'):
                pfcp_info.append("HEARTBEAT ONLY")
            parts.append(" | ".join(pfcp_info))
        
        if features.get('ngap_packets', 0) > 0:
            parts.append(f"NGAP packets: {features['ngap_packets']}")
        
        # Add TCP/UDP port information
        if features.get('tcp_ports'):
            parts.append(f"TCP ports: {', '.join(map(str, sorted(features['tcp_ports'])[:10]))}" + 
                        ("..." if len(features['tcp_ports']) > 10 else ""))
                        
        if features.get('udp_ports'):
            parts.append(f"UDP ports: {', '.join(map(str, sorted(features['udp_ports'])[:10]))}" + 
                        ("..." if len(features['udp_ports']) > 10 else ""))
        
        # ICMP echo stats to distinguish downlink failures
        if features.get('protocol_counts', {}).get('ICMP', 0) > 0:
            parts.append(
                f"ICMP echo req/reply: {features.get('icmp_echo_request_count', 0)}/" \
                f"{features.get('icmp_echo_reply_count', 0)}"
            )
        
        return ". ".join(parts)
