#!/usr/bin/env python3
"""Command-line interface for the 5G PCAP Analysis Tool."""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

from .modules.data_prep.pcap_processor import PCAPProcessor
from .modules.model_training.trainer import PCAPModelTrainer
from .modules.testing.predictor import PCAPPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

class PCAPAnalyzerCLI:
    """Command-line interface for the 5G PCAP Analysis Tool."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the CLI with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_directories()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        dirs = [
            self.config['data']['raw_pcaps'],
            self.config['data']['processed'],
            self.config['data']['embeddings'],
            self.config['data']['models'],
            self.config['data']['results']
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def process_pcaps(self, input_dir: str, output_file: str, mapping_file: str = None) -> None:
        """Process PCAP files and extract features.
        
        Args:
            input_dir: Directory containing PCAP files
            output_file: Path to save processed features
            mapping_file: Optional path to CSV file mapping PCAP filenames to labels
        """
        logger.info(f"Processing PCAPs from {input_dir}")
        
                
        processor = PCAPProcessor(self.config, mapping_file=mapping_file)
        processor.process_directory(input_dir, output_file)
    
    def train_model(self, features_file: str, output_dir: str) -> None:
        """Train the classification model.
        
        Args:
            features_file: Path to the JSON file with extracted features
            output_dir: Directory to save the trained model
        """
        logger.info("Starting model training...")
        trainer = PCAPModelTrainer(self.config)
        
        # Load and prepare data
        X_train, y_train, X_test, y_test = trainer.load_data(features_file)
        
        # Train model
        trainer.train_model(X_train, y_train)
        
        # Evaluate model
        metrics = trainer.evaluate_model(X_test, y_test)
        logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
        
        # Build vector database for RAG
        trainer.build_vector_db(features_file, os.path.join(output_dir, 'vector_db'))
        
        # Save model
        trainer.save_model(output_dir)
        logger.info(f"Model and vector database saved to {output_dir}")
    
    def predict(self, pcap_path: str, model_dir: str, output_file: Optional[str] = None) -> Dict:
        """Classify a PCAP file and provide explanation.
        
        Args:
            pcap_path: Path to the PCAP file
            model_dir: Directory containing the trained model and vector database
            output_file: Optional path to save the prediction results
            
        Returns:
            Dictionary containing prediction results
        """
        logger.info(f"Analyzing PCAP: {pcap_path}")
        
        # Initialize predictor
        model_path = os.path.join(model_dir, 'pcap_classifier.joblib')
        vector_db_dir = os.path.join(model_dir, 'vector_db')
        predictor = PCAPPredictor(model_path, vector_db_dir, self.config)
        
        # Make prediction
        result = predictor.predict(pcap_path)
        
        # Print results
        self._print_prediction(result)
        
        # Save results if output file is specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return result
    
    def _print_prediction(self, result: Dict) -> None:
        """Print prediction results in a human-readable format.
        
        Args:
            result: Prediction results dictionary
        """
        print("\n" + "="*50)
        print(f"PCAP Analysis Report: {result['pcap_name']}")
        print("="*50)
        
        # Print prediction
        pred = "FAILURE" if result['prediction'] == 1 else "SUCCESS"
        confidence = result['confidence'] * 100
        print(f"\nPrediction: {pred} (Confidence: {confidence:.1f}%)")
        
        # Print key indicators
        if 'explanation' in result and 'key_indicators' in result['explanation']:
            print("\nKey Indicators:")
            for indicator in result['explanation']['key_indicators']:
                prefix = "[!] " if indicator['type'] == 'warning' else "[i] "
                print(f"{prefix}{indicator['message']}")
        
        # Print summary
        if 'explanation' in result and 'summary' in result['explanation']:
            print("\nSummary:")
            print(result['explanation']['summary'])
        
        # Print similar examples
        if 'explanation' in result and 'similar_examples' in result['explanation']:
            print("\nSimilar Examples:")
            for i, example in enumerate(result['explanation']['similar_examples'], 1):
                print(f"\n  Example {i}:")
                print(f"  - File: {example['pcap_name']}")
                print(f"  - Similarity: {example['similarity']:.2f}")
                print(f"  - Packets: {example['total_packets']}")
                print(f"  - Errors: {example['error_count']}")
        
        print("\n" + "="*50 + "\n")

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="5G PCAP Analysis Tool")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process PCAP files and extract features')
    process_parser.add_argument('input_dir', help='Directory containing PCAP files')
    process_parser.add_argument('--output', required=True, help='Path to save processed features')
    process_parser.add_argument('--mapping', help='Path to CSV file mapping PCAP filenames to labels')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the classification model')
    train_parser.add_argument('features_file', help='JSON file with extracted features')
    train_parser.add_argument('--output-dir', '-o', default='models', 
                            help='Directory to save the trained model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Classify a PCAP file')
    predict_parser.add_argument('pcap_file', help='Path to the PCAP file to analyze')
    predict_parser.add_argument('--model-dir', '-m', default='models',
                              help='Directory containing the trained model')
    predict_parser.add_argument('--output', '-o', 
                              help='Output JSON file for prediction results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    if args.command == 'process':
        cli = PCAPAnalyzerCLI()
        cli.process_pcaps(args.input_dir, args.output, args.mapping)
    elif args.command == 'train':
        cli = PCAPAnalyzerCLI()
        cli.train_model(args.features_file, args.output_dir)
    elif args.command == 'predict':
        cli = PCAPAnalyzerCLI()
        result = cli.predict(args.pcap_file, args.model_dir, args.output)
        cli._print_prediction(result)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
