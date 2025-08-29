import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib
import faiss
from sentence_transformers import SentenceTransformer

from ..data_prep.pcap_processor import PCAPProcessor
from ..data_prep.pfcp_cause_codes import get_pfcp_cause_analyzer

class PCAPPredictor:
    """Classify new PCAP files and provide explanations using RAG."""
    
    def __init__(self, model_path: str, vector_db_dir: str, config: dict):
        """Initialize the predictor with trained model and vector database.
        
        Args:
            model_path: Path to the trained model file
            vector_db_dir: Directory containing the vector database files
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = joblib.load(model_path)
        self.logger.info(f"Loaded model from {model_path}")
        
        # Load vector database
        self.vector_db = faiss.read_index(os.path.join(vector_db_dir, 'vector_db.index'))
        with open(os.path.join(vector_db_dir, 'metadata.json'), 'r') as f:
            self.vector_db_metadata = json.load(f)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger.info(f"Loaded vector database with {len(self.vector_db_metadata)} entries")
        
        # Initialize PCAP processor
        self.pcap_processor = PCAPProcessor(config)
        
        # Initialize PFCP cause analyzer
        self.pfcp_analyzer = get_pfcp_cause_analyzer()
    
    def predict(self, pcap_path: str) -> Dict[str, Any]:
        """Classify a PCAP file and provide explanation.
        
        Args:
            pcap_path: Path to the PCAP file
            
        Returns:
            Dictionary containing prediction and explanation
        """
        try:
            # Process PCAP to extract features
            features = self.pcap_processor.process_pcap(pcap_path)
            
            # Prepare features for prediction
            X = self._prepare_features(features)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            # Get prediction probabilities if available
            proba = self.model.predict_proba(X)[0] if hasattr(self.model, 'predict_proba') else [1.0]
            
            # Handle single-class case
            if len(proba) == 1:
                class_probs = {
                    'success': float(proba[0]),
                    'failure': 1.0 - float(proba[0])  # Calculate complement for binary case
                }
                # For single-class models, use the probability directly as confidence
                confidence = float(proba[0])
            else:
                class_probs = {
                    'success': float(proba[0]),
                    'failure': float(proba[1])
                }
                confidence = max(proba)
            
            # Get explanation using RAG
            explanation = self._explain_prediction(features)
            
            return {
                'pcap_name': os.path.basename(pcap_path),
                'prediction': int(prediction),
                'confidence': confidence,
                'class_probabilities': class_probs,
                'features': features,
                'explanation': explanation,
                'model_type': 'single_class' if len(proba) == 1 else 'multi_class'
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting {pcap_path}: {str(e)}")
            raise
    
    def _prepare_features(self, features: Dict) -> pd.DataFrame:
        """Prepare features for model prediction.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            DataFrame with features in expected format
        """
        X = pd.DataFrame([{
            'total_packets': features['total_packets'],
            'avg_packet_size': features['avg_packet_size'],
            'avg_timing': features['avg_timing'],
            'error_count': features['error_count'],
            'ngap_message_count': features['ngap_message_count'],
            'count_TCP': features['protocol_counts']['TCP'],
            'count_UDP': features['protocol_counts']['UDP'],
            'count_SCTP': features['protocol_counts']['SCTP']
        }])
        return X
    
    def _explain_prediction(self, features: Dict, k: int = 3) -> Dict:
        """Generate explanation for prediction using RAG.
        
        Args:
            features: Dictionary of extracted features
            k: Number of similar examples to retrieve
            
        Returns:
            Dictionary containing explanation and similar examples
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode([features['description']])
        
        # Search for similar examples
        distances, indices = self.vector_db.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        # Get similar examples
        similar_examples = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0:  # FAISS returns -1 for invalid indices
                example = self.vector_db_metadata[idx]
                similar_examples.append({
                    'pcap_name': example['pcap_name'],
                    'similarity': float(1 / (1 + distance)),  # Convert distance to similarity
                    'total_packets': example['total_packets'],
                    'error_count': example['error_count'],
                    'description': example['features'].get('description', '')
                })
        
        # Generate explanation
        explanation = {
            'summary': self._generate_summary(features, similar_examples),
            'similar_examples': similar_examples,
            'key_indicators': self._get_key_indicators(features)
        }
        
        return explanation
    
    def _generate_summary(self, features: Dict, similar_examples: List[Dict]) -> str:
        """Generate a summary explanation.
        
        Args:
            features: Extracted features
            similar_examples: List of similar examples
            
        Returns:
            String summary
        """
        parts = []
        
        # Basic info
        parts.append(f"The PCAP contains {features['total_packets']} packets with "
                    f"{features['error_count']} potential error(s).")
        
        # Protocol info
        protocols = ", ".join([f"{k} ({v} packets)" for k, v in features['protocol_counts'].items()])
        parts.append(f"Protocol distribution: {protocols}.")
        
        # Similar examples
        if similar_examples:
            similar_errors = sum(1 for ex in similar_examples if ex['error_count'] > 0)
            parts.append(
                f"Found {len(similar_examples)} similar PCAPs in the database, "
                f"of which {similar_errors} contained errors."
            )
        
        return " ".join(parts)
    
    def _get_key_indicators(self, features: Dict) -> List[Dict]:
        """Extract key indicators from features.
        
        Args:
            features: Extracted features
            
        Returns:
            List of key indicators
        """
        indicators = []
        
        # Error indicators
        if features['error_count'] > 0:
            indicators.append({
                'type': 'warning',
                'message': f"Found {features['error_count']} potential error(s) in the PCAP."
            })
        
        # PFCP indicators (N4 control plane)
        if features.get('pfcp_packets', 0) > 0:
            # Message types / causes
            pfcp_types = features.get('pfcp_message_types', [])
            pfcp_causes = features.get('pfcp_cause_codes', [])
            if pfcp_types:
                indicators.append({
                    'type': 'info',
                    'message': f"PFCP observed: message_types={pfcp_types}"
                })
            if pfcp_causes:
                # Use comprehensive cause code analysis
                cause_analysis = self.pfcp_analyzer.analyze_cause_codes(pfcp_causes)
                cause_summary = self.pfcp_analyzer.get_cause_summary_text(pfcp_causes)
                
                indicators.append({
                    'type': cause_analysis['severity'],
                    'message': f"PFCP causes: {cause_summary}"
                })
                
                # Add specific insights from cause analysis
                for insight in cause_analysis['insights']:
                    indicators.append({
                        'type': cause_analysis['severity'],
                        'message': f"PFCP Analysis: {insight}"
                    })
            
            # Enhanced failure detection
            if features.get('pfcp_session_establishment_failed'):
                indicators.append({
                    'type': 'warning',
                    'message': "PFCP Session Establishment Response indicates FAILURE"
                })
            if features.get('pfcp_session_modification_failed'):
                indicators.append({
                    'type': 'warning',
                    'message': "PFCP Session Modification Response indicates FAILURE"
                })
            if features.get('pfcp_session_deletion_failed'):
                indicators.append({
                    'type': 'warning',
                    'message': "PFCP Session Deletion Response indicates FAILURE"
                })
            if features.get('pfcp_session_report_failed'):
                indicators.append({
                    'type': 'warning',
                    'message': "PFCP Session Report Response indicates FAILURE"
                })
            if features.get('pfcp_heartbeat_only'):
                indicators.append({
                    'type': 'info',
                    'message': "PFCP heartbeat-only traffic detected (no session procedures)."
                })
        
        # GTP-U indicators (user plane)
        if features.get('gtp_packets', 0) > 0:
            inner = features.get('gtp_inner_protocols', [])
            indicators.append({
                'type': 'info',
                'message': f"GTP-U observed: inner_protocols={inner}, icmp_in_gtp={features.get('gtp_icmp_packets', 0)}"
            })
            # Highlight missing ICMP replies inside GTP-U
            if 1 in inner:
                req = int(features.get('icmp_echo_request_count', 0))
                rep = int(features.get('icmp_echo_reply_count', 0))
                if req > rep:
                    missing = req - rep
                    indicators.append({
                        'type': 'warning',
                        'message': f"ICMP echo replies missing in GTP-U: {missing} unmatched request(s). Possible downlink path or return routing issue."
                    })
                elif req == 0 and features.get('gtp_icmp_packets', 0) > 0:
                    indicators.append({
                        'type': 'info',
                        'message': "ICMP seen in GTP-U but no echo requests counted (non-echo ICMP or parsing edge case)."
                    })
        
        # Protocol indicators for NGAP (only if no PFCP/GTP present)
        has_pfcp_or_gtp = features.get('pfcp_packets', 0) > 0 or features.get('gtp_packets', 0) > 0
        if not has_pfcp_or_gtp and features['protocol_counts']['SCTP'] == 0 and features['ngap_message_count'] == 0:
            indicators.append({
                'type': 'info',
                'message': "No SCTP traffic or NGAP messages detected. This might not be 5G control plane traffic."
            })
        
        # Timing indicators
        if features.get('avg_timing', 0) > 1.0:  # More than 1 second average delay
            indicators.append({
                'type': 'warning',
                'message': f"High average inter-packet delay: {features['avg_timing']:.2f}s"
            })
        
        return indicators
