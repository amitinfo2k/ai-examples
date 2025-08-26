"""
Vector store implementation using FAISS for efficient similarity search.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import numpy as np
import faiss
import os

class VectorStore:
    """
    A vector store for storing and querying PCAP feature embeddings.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(self, dimension: int = 384, index_path: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embeddings
            index_path: Path to load/save the FAISS index (optional)
        """
        self.dimension = dimension
        self.index_path = index_path
        self.metadata = []
        
        # Initialize FAISS index
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            # Load metadata
            metadata_path = self._get_metadata_path(index_path)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
        else:
            # Create a new index
            self.index = faiss.IndexFlatL2(dimension)
    
    def add_embedding(
        self, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ) -> int:
        """
        Add an embedding and its metadata to the store.
        
        Args:
            embedding: The embedding vector to add
            metadata: Dictionary of metadata associated with the embedding
            
        Returns:
            The index of the added embedding
        """
        # Convert to numpy array and reshape for FAISS
        vector = np.array(embedding, dtype='float32').reshape(1, -1)
        
        # Add to index
        self.index.add(vector)
        
        # Store metadata
        self.metadata.append(metadata)
        
        return len(self.metadata) - 1
    
    def search(
        self, 
        query_embedding: List[float], 
        k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: The query embedding
            k: Number of results to return
            
        Returns:
            List of (metadata, distance) tuples, sorted by distance
        """
        if len(self.metadata) == 0:
            return []
            
        # Convert query to numpy array and reshape for FAISS
        query = np.array(query_embedding, dtype='float32').reshape(1, -1)
        
        # Search the index
        distances, indices = self.index.search(query, min(k, len(self.metadata)))
        
        # Get results with metadata
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):  # Ensure index is valid
                results.append((self.metadata[idx], float(dist)))
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the index and metadata to disk.
        
        Args:
            path: Path to save the index (defaults to the path provided at init)
        """
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No path provided to save the index")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, save_path)
        
        # Save metadata
        metadata_path = self._get_metadata_path(save_path)
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def _get_metadata_path(self, index_path: str) -> str:
        """Get the path for the metadata file based on the index path."""
        return f"{index_path}.meta"
    
    def __len__(self) -> int:
        """Get the number of vectors in the store."""
        return len(self.metadata)
    
    @classmethod
    def load(cls, path: str) -> 'VectorStore':
        """
        Load a VectorStore from disk.
        
        Args:
            path: Path to the saved index
            
        Returns:
            Loaded VectorStore instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
            
        # Create a new instance with the path
        store = cls(index_path=path)
        return store
