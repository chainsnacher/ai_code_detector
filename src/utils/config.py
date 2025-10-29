"""
Configuration management for the AI Code Detection System.
Handles loading and validation of configuration settings.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class Config:
    """Centralized configuration management."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._setup_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _validate_config(self):
        """Validate configuration structure and values."""
        required_sections = [
            'data_collection', 'features', 'models', 'database', 
            'web_app', 'evaluation', 'logging'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            'data/raw', 'data/processed', 'data/train', 'data/test', 'data/validation',
            'models', 'results', 'logs', 'web_app/static', 'web_app/templates'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'models.baseline.random_forest.n_estimators')."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def update(self, key: str, value: Any):
        """Update configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = path or self.config_path
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance."""
    return config
