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
            parts.append(f"GTP packets: {features['gtp_packets']}")
            parts.append(f"GTP TEIDs: {len(features.get('gtp_teids', []))} unique")
            
        if features.get('pfcp_packets', 0) > 0:
            parts.append(f"PFCP packets: {features['pfcp_packets']}")
            
        if features.get('ngap_packets', 0) > 0:
            parts.append(f"NGAP packets: {features['ngap_packets']}")
        
        # Add TCP/UDP port information
        if features.get('tcp_ports'):
            parts.append(f"TCP ports: {', '.join(map(str, sorted(features['tcp_ports'])[:10]))}" + 
                        ("..." if len(features['tcp_ports']) > 10 else ""))
                        
        if features.get('udp_ports'):
            parts.append(f"UDP ports: {', '.join(map(str, sorted(features['udp_ports'])[:10]))}" + 
                        ("..." if len(features['udp_ports']) > 10 else ""))
        
        return ". ".join(parts)
