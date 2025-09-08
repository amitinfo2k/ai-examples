import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sentence_transformers import SentenceTransformer
import faiss

class PCAPModelTrainer:
    """Train and manage models for PCAP classification."""
    
    def __init__(self, config: dict):
        """Initialize the model trainer with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db = None
        self.vector_db_metadata = []
        
    def load_data(self, features_file: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and prepare training data from features file.
        
        Args:
            features_file: Path to JSON file containing extracted features
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        with open(features_file, 'r') as f:
            features_list = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # For now, we'll use a simple binary classification
        # In a real scenario, you'd have proper labels in your data
        # Here we'll simulate it based on error count
        df['label'] = (df['error_count'] > 0).astype(int)
        
        # Extract features for training
        X = pd.DataFrame()
        X['total_packets'] = df['total_packets']
        X['avg_packet_size'] = df['avg_packet_size']
        X['avg_timing'] = df['avg_timing']
        X['error_count'] = df['error_count']
        X['ngap_message_count'] = df['ngap_message_count']
        
        # Add protocol counts
        for proto in ['TCP', 'UDP', 'SCTP']:
            X[f'count_{proto}'] = df['protocol_counts'].apply(lambda x: x[proto])
        
        y = df['label']
        
        # Check if we have enough samples for stratified split
        from collections import Counter
        label_counts = Counter(y)
        min_class_count = min(label_counts.values())
        
        # Split into train and test sets
        if min_class_count >= 2 and len(y) >= 4:  # Need at least 2 per class and 4 total for meaningful split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # For small datasets, use simple random split without stratification
            print(f"Warning: Small dataset detected (min class count: {min_class_count}, total: {len(y)}). Using simple random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the classification model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.logger.info("Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed.")
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        y_pred = self.model.predict(X_test)
        
        # Check if we have only one class in the test set
        unique_classes = set(y_test.unique())
        single_class = len(unique_classes) == 1
        
        # Only calculate probabilities if we have multiple classes
        y_proba = None
        if not single_class:
            try:
                y_proba = self.model.predict_proba(X_test)[:, 1]
            except (IndexError, AttributeError):
                self.logger.warning("Could not calculate prediction probabilities")
                y_proba = None
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'single_class': single_class,
            'classes_present': list(unique_classes)
        }
        
        # Only include feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            metrics['feature_importances'] = dict(zip(X_test.columns, self.model.feature_importances_))
        
        if single_class:
            self.logger.warning(f"Only one class present in test set: {unique_classes}")
        self.logger.info(f"Model evaluation complete. Accuracy: {accuracy:.4f}")
        return metrics
    
    def build_vector_db(self, features_file: str, output_dir: str) -> None:
        """Build a vector database for RAG from PCAP features.
        
        Args:
            features_file: Path to JSON file containing extracted features
            output_dir: Directory to save the vector database
        """
        with open(features_file, 'r') as f:
            features_list = json.load(f)
        
        # Extract descriptions and metadata
        descriptions = [item['description'] for item in features_list]
        self.vector_db_metadata = [
            {
                'pcap_name': item['pcap_name'],
                'total_packets': item['total_packets'],
                'error_count': item['error_count'],
                'features': item
            }
            for item in features_list
        ]
        
        # Generate embeddings
        self.logger.info("Generating embeddings for vector database...")
        embeddings = self.embedding_model.encode(descriptions, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_db = faiss.IndexFlatL2(dimension)
        self.vector_db.add(np.array(embeddings).astype('float32'))
        
        # Save vector database and metadata
        os.makedirs(output_dir, exist_ok=True)
        faiss.write_index(self.vector_db, os.path.join(output_dir, 'vector_db.index'))
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.vector_db_metadata, f)
            
        self.logger.info(f"Vector database created with {len(embeddings)} entries.")
    
    def save_model(self, output_dir: str) -> None:
        """Save the trained model and related artifacts.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model has been trained yet.")
            
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'pcap_classifier.joblib')
        joblib.dump(self.model, model_path)
        self.logger.info(f"Model saved to {model_path}")
