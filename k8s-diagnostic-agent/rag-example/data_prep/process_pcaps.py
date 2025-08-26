"""
Main script for processing PCAP files and generating embeddings.
"""
import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import load_config, PCAPConfig
from .pcap_processor import PCAPProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore

def process_pcap(
    pcap_path: str,
    config: PCAPConfig,
    pcap_processor: PCAPProcessor,
    embedder: EmbeddingGenerator,
    vector_store: VectorStore,
    base_dir: str = ""
) -> Dict[str, Any]:
    """
    Process a single PCAP file and add its embedding to the vector store.
    
    Args:
        pcap_path: Path to the PCAP file
        config: PCAP configuration
        pcap_processor: PCAPProcessor instance
        embedder: EmbeddingGenerator instance
        vector_store: VectorStore instance
        base_dir: Base directory for relative paths
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Handle relative paths
        full_path = os.path.join(base_dir, pcap_path) if base_dir else pcap_path
        
        # Process PCAP file
        print(f"Processing {full_path}...")
        features = pcap_processor.process_pcap(full_path)
        
        # Generate feature text and embedding
        feature_text = embedder._features_to_text(features)
        feature_embedding = embedder.generate_embedding(feature_text)
        
        # Generate description embedding
        description_embedding = embedder.generate_embedding(config.description)
        
        # Combine embeddings (simple average)
        combined_embedding = (
            np.array(feature_embedding) + np.array(description_embedding)
        ) / 2.0
        
        # Prepare metadata
        metadata = {
            'file': pcap_path,
            'label': config.label,
            'issue_type': config.issue_type,
            'description': config.description,
            'key_patterns': config.key_patterns,
            'features': features,
            'feature_text': feature_text,
        }
        
        # Add to vector store
        vector_store.add_embedding(combined_embedding.tolist(), metadata)
        
        return {
            'status': 'success',
            'file': pcap_path,
            'num_packets': features.get('total_packets', 0),
            'features': {k: v for k, v in features.items() if not isinstance(v, (list, dict, set))}
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'file': pcap_path,
            'error': str(e)
        }

def main():
    """Main function to process PCAPs and generate embeddings."""
    parser = argparse.ArgumentParser(description='Process PCAP files and generate embeddings.')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to the configuration JSON file')
    parser.add_argument('--output', type=str, default='vector_store.faiss',
                       help='Output path for the FAISS index')
    parser.add_argument('--base-dir', type=str, default='',
                       help='Base directory for PCAP file paths')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        configs = load_config(args.config)
        print(f"Loaded configuration with {len(configs)} PCAP entries")
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        return 1
    
    # Initialize components
    pcap_processor = PCAPProcessor()
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(dimension=embedder.embedding_dim)
    
    # Process each PCAP file
    results = []
    for config in configs:
        result = process_pcap(
            pcap_path=config.file,
            config=config,
            pcap_processor=pcap_processor,
            embedder=embedder,
            vector_store=vector_store,
            base_dir=args.base_dir
        )
        results.append(result)
    
    # Save vector store
    try:
        vector_store.save(args.output)
        print(f"Saved vector store to {args.output} with {len(vector_store)} entries")
    except Exception as e:
        print(f"Error saving vector store: {str(e)}")
        return 1
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    print("\nProcessing complete!")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    
    if error_count > 0:
        print("\nErrors:")
        for result in results:
            if result['status'] == 'error':
                print(f"- {result['file']}: {result['error']}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
