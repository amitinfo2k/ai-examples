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
            
            # Enhanced prediction override based on NGAP failure analysis
            # Only override when we have explicit unsuccessfulOutcome or NAS-level reject evidence
            # Only treat as explicit reject for specific reject/failure procedures
            explicit_ngap_reject = False
            if features.get('ngap_messages'):
                for msg in features['ngap_messages']:
                    proc = msg.get('procedure_code')
                    nas_type = msg.get('nas_message_type')
                    msg_type = msg.get('message_type')
                    if nas_type in ('RegistrationReject', 'SecurityModeReject'):
                        explicit_ngap_reject = True
                        break
                    # NGAP unsuccessfulOutcome for specific failure procedures only
                    if msg_type == 2 and proc in (5, 13, 16, 18, 770, 775, 780, 783, 785):
                        explicit_ngap_reject = True
                        break

            if explicit_ngap_reject or features.get('ngap_registration_status') == 'failed' or features.get('protocol_handshake_completion', {}).get('ngap_initial_context_setup') == 'failed':
                prediction = 1
                class_probs = {'success': 0.0, 'failure': 1.0}
                confidence = 1.0
            
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
        # Use only the original features that the model was trained with
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
        
        # Enhanced NGAP indicators (5G control plane)
        if features.get('ngap_message_count', 0) > 0:
            # Procedure types (show unique procedures with counts)
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
                
                indicators.append({
                    'type': 'info',
                    'message': f"NGAP procedures detected: {', '.join(procedure_summary)}"
                })
            
            # UE Setup vs gNB Setup differentiation
            if features.get('ngap_messages'):
                ue_setup_count = sum(1 for msg in features['ngap_messages'] if msg.get('is_ue_setup', False))
                gnb_setup_count = sum(1 for msg in features['ngap_messages'] if msg.get('is_gnb_setup', False))
                
                if ue_setup_count > 0:
                    indicators.append({
                        'type': 'info',
                        'message': f"UE Setup procedures detected: {ue_setup_count} message(s)"
                    })
                
                if gnb_setup_count > 0:
                    indicators.append({
                        'type': 'info',
                        'message': f"gNB Setup procedures detected: {gnb_setup_count} message(s)"
                    })
            
            # Registration status
            if features.get('ngap_registration_status'):
                status = features['ngap_registration_status']
                if status == 'success':
                    indicators.append({
                        'type': 'success',
                        'message': f"NGAP registration completed successfully"
                    })
                elif status == 'failed':
                    indicators.append({
                        'type': 'error',
                        'message': f"NGAP registration failed"
                    })
                elif status == 'partial':
                    indicators.append({
                        'type': 'warning',
                        'message': f"NGAP registration partially completed - may indicate incomplete procedure"
                    })
                elif status == 'unknown':
                    indicators.append({
                        'type': 'warning',
                        'message': f"NGAP registration status unknown - insufficient procedure information"
                    })
            
            # Authentication and security steps
            auth_steps = len(features.get('ngap_authentication_steps', []))
            security_steps = len(features.get('ngap_security_steps', []))
            if auth_steps > 0:
                indicators.append({
                    'type': 'info',
                    'message': f"NGAP authentication steps: {auth_steps} completed"
                })
            if security_steps > 0:
                indicators.append({
                    'type': 'info',
                    'message': f"NGAP security setup steps: {security_steps} completed"
                })
            
            # Cause codes for failures
            if features.get('ngap_detailed_causes'):
                # Prefer human-readable detailed causes if available
                from ..data_prep.ngap_cause_codes import get_ngap_cause_text
                text_counts = {}
                for dc in features['ngap_detailed_causes']:
                    label = get_ngap_cause_text(dc.get('category'), dc.get('value'))
                    text_counts[label] = text_counts.get(label, 0) + 1
                formatted = []
                for label, count in sorted(text_counts.items()):
                    formatted.append(f"{label} (x{count})" if count > 1 else label)
                indicators.append({'type': 'error', 'message': f"NGAP failure causes: {', '.join(formatted)}"})
            elif features.get('ngap_cause_codes'):
                # Fallback to numeric codes aggregation
                cause_counts = {}
                for cause in features['ngap_cause_codes']:
                    cause_counts[str(cause)] = cause_counts.get(str(cause), 0) + 1
                formatted_causes = []
                for cause_str, count in sorted(cause_counts.items(), key=lambda x: int(x[0])):
                    formatted_causes.append(f"{cause_str} (x{count})" if count > 1 else cause_str)
                indicators.append({'type': 'error', 'message': f"NGAP failure cause codes: {', '.join(formatted_causes)}"})
            
            # NGAP Setup and Initial Context Setup specific indicators
            if features.get('protocol_handshake_completion'):
                handshake_status = features['protocol_handshake_completion']
                
                # NGAP Setup indicators
                if 'ngap_setup' in handshake_status:
                    setup_status = handshake_status['ngap_setup']
                    if setup_status == 'failed':
                        indicators.append({
                            'type': 'error',
                            'message': "NGAP Setup FAILED - critical procedure failure detected"
                        })
                    elif setup_status == 'incomplete':
                        indicators.append({
                            'type': 'warning',
                            'message': "NGAP Setup incomplete - missing response or timeout"
                        })
                    elif setup_status == 'complete':
                        indicators.append({
                            'type': 'success',
                            'message': "NGAP Setup completed successfully"
                        })
                
                # Initial Context Setup indicators
                if 'ngap_initial_context_setup' in handshake_status:
                    setup_status = handshake_status['ngap_initial_context_setup']
                    if setup_status == 'failed':
                        indicators.append({
                            'type': 'error',
                            'message': "NGAP Initial Context Setup FAILED - critical procedure failure detected"
                        })
                    elif setup_status == 'incomplete':
                        indicators.append({
                            'type': 'warning',
                            'message': "NGAP Initial Context Setup incomplete - missing response or timeout"
                        })
                    elif setup_status == 'complete':
                        indicators.append({
                            'type': 'success',
                            'message': "NGAP Initial Context Setup completed successfully"
                        })
                
                if 'ngap_authentication' in handshake_status:
                    auth_status = handshake_status['ngap_authentication']
                    if auth_status == 'incomplete':
                        indicators.append({
                            'type': 'warning',
                            'message': "NGAP Authentication incomplete - missing response or timeout"
                        })
                
                if 'ngap_security' in handshake_status:
                    security_status = handshake_status['ngap_security']
                    if security_status == 'failed':
                        indicators.append({
                            'type': 'error',
                            'message': "NGAP Security Mode REJECTED - security configuration failure"
                        })
                    elif security_status == 'incomplete':
                        indicators.append({
                            'type': 'warning',
                            'message': "NGAP Security Mode incomplete - missing response or timeout"
                        })
        
        # Protocol indicators for NGAP (only if no PFCP/GTP present)
        has_pfcp_or_gtp = features.get('pfcp_packets', 0) > 0 or features.get('gtp_packets', 0) > 0
        if not has_pfcp_or_gtp and features['protocol_counts']['SCTP'] == 0 and features['ngap_message_count'] == 0:
            indicators.append({
                'type': 'info',
                'message': "No SCTP traffic or NGAP messages detected. This might not be 5G control plane traffic."
            })
        
        # Enhanced failure pattern indicators
        if features.get('has_failures'):
            if features.get('failure_patterns'):
                patterns = features['failure_patterns']
                indicators.append({
                    'type': 'error',
                    'message': f"Failure patterns detected: {', '.join(patterns)}"
                })
            
            if features.get('failure_scenarios'):
                scenarios = features['failure_scenarios']
                indicators.append({
                    'type': 'error',
                    'message': f"Failure scenarios: {', '.join(scenarios)}"
                })
            
            if features.get('root_cause_indicators'):
                root_causes = features['root_cause_indicators']
                indicators.append({
                    'type': 'error',
                    'message': f"Root cause indicators: {', '.join(root_causes)}"
                })
        
        # Enhanced timing and sequence indicators
        if features.get('timing_anomalies'):
            anomaly_count = len(features['timing_anomalies'])
            indicators.append({
                'type': 'warning',
                'message': f"Timing anomalies detected: {anomaly_count} unusual delays"
            })
        
        if features.get('sequence_anomalies'):
            seq_count = len(features['sequence_anomalies'])
            indicators.append({
                'type': 'warning',
                'message': f"Sequence anomalies detected: {seq_count} out-of-order packets"
            })
        
        if features.get('retransmission_patterns'):
            retrans_count = len(features['retransmission_patterns'])
            indicators.append({
                'type': 'warning',
                'message': f"Retransmission indicators: {retrans_count} potential retransmissions"
            })
        
        # Enhanced PFCP indicators
        if features.get('pfcp_association_status'):
            status = features['pfcp_association_status']
            if status == 'failed':
                indicators.append({
                    'type': 'error',
                    'message': "PFCP association failed"
                })
            elif status == 'established':
                indicators.append({
                    'type': 'success',
                    'message': "PFCP association established successfully"
                })
        
        if features.get('pfcp_session_establishment_success_rate', 0) > 0:
            rate = features['pfcp_session_establishment_success_rate']
            if rate < 1.0:
                indicators.append({
                    'type': 'warning',
                    'message': f"PFCP session establishment success rate: {rate:.2f} ({rate*100:.1f}%)"
                })
        
        # Enhanced GTP indicators
        if features.get('gtp_tunnel_count', 0) > 0:
            indicators.append({
                'type': 'info',
                'message': f"GTP tunnels detected: {features['gtp_tunnel_count']} unique TEIDs"
            })
        
        if features.get('gtp_user_plane_flows') and features.get('gtp_control_plane_messages'):
            user_flows = len(features['gtp_user_plane_flows'])
            control_msgs = len(features['gtp_control_plane_messages'])
            indicators.append({
                'type': 'info',
                'message': f"GTP traffic: {user_flows} user plane flows, {control_msgs} control messages"
            })
        
        # Timing indicators
        if features.get('avg_timing', 0) > 1.0:  # More than 1 second average delay
            indicators.append({
                'type': 'warning',
                'message': f"High average inter-packet delay: {features['avg_timing']:.2f}s"
            })
        
        return indicators
    
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
        return procedure_names.get(procedure_code, f'UnknownProcedure_{procedure_code}')
