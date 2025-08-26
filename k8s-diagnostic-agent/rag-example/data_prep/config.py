"""
Configuration loader for the data preparation module.
Handles loading and validating the configuration from JSON files.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class PCAPConfig:
    """Configuration for a single PCAP file."""
    file: str
    label: int
    issue_type: str
    description: str
    key_patterns: List[str]

class ConfigLoader:
    """Loads and validates configuration from JSON files."""
    
    @staticmethod
    def load_config(config_path: str) -> List[PCAPConfig]:
        """
        Load and validate configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration JSON file
            
        Returns:
            List of PCAPConfig objects
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config is not valid JSON
            ValueError: If config is missing required fields
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        if 'input' not in config_data:
            raise ValueError("Config must contain an 'input' key with a list of PCAP configurations")
            
        pcap_configs = []
        for item in config_data['input']:
            # Validate required fields
            required_fields = ['file', 'label', 'issue_type', 'description', 'key_patterns']
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing required field in config: {field}")
                    
            pcap_configs.append(PCAPConfig(
                file=item['file'],
                label=item['label'],
                issue_type=item['issue_type'],
                description=item['description'],
                key_patterns=item.get('key_patterns', [])
            ))
            
        return pcap_configs

def load_config(config_path: str) -> List[PCAPConfig]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        List of PCAPConfig objects
    """
    return ConfigLoader.load_config(config_path)
