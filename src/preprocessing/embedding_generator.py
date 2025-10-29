"""
Advanced code embedding generation using multiple models.
Generates contextual embeddings using CodeBERT, GraphCodeBERT, and custom models.
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModel, 
    RobertaTokenizer, RobertaModel
)
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger("ai_code_detector")

class CodeEmbeddingGenerator:
    """Generates multiple types of code embeddings."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize embedding generator with configuration."""
        self.config = config or {
            'models': ['codebert', 'graphcodebert'],
            'embedding_dim': 768,
            'max_length': 512,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        self.models = {}
        self.tokenizers = {}
        self.device = self.config['device']
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models for embedding generation."""
        # Load each requested model independently; skip on failure
        if 'codebert' in self.config['models']:
            try:
                self.tokenizers['codebert'] = AutoTokenizer.from_pretrained('microsoft/codebert-base')
                self.models['codebert'] = AutoModel.from_pretrained('microsoft/codebert-base')
                self.models['codebert'].to(self.device)
                logger.info("CodeBERT model loaded successfully")
            except Exception as e:
                logger.warning(f"Skipping CodeBERT due to error: {e}")

        if 'graphcodebert' in self.config['models']:
            try:
                self.tokenizers['graphcodebert'] = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
                self.models['graphcodebert'] = AutoModel.from_pretrained('microsoft/graphcodebert-base')
                self.models['graphcodebert'].to(self.device)
                logger.info("GraphCodeBERT model loaded successfully")
            except Exception as e:
                logger.warning(f"Skipping GraphCodeBERT due to error: {e}")

        if 'roberta' in self.config['models']:
            try:
                self.tokenizers['roberta'] = RobertaTokenizer.from_pretrained('roberta-base')
                self.models['roberta'] = RobertaModel.from_pretrained('roberta-base')
                self.models['roberta'].to(self.device)
                logger.info("RoBERTa model loaded successfully")
            except Exception as e:
                logger.warning(f"Skipping RoBERTa due to error: {e}")
    
    def generate_embeddings(self, code: str, model_name: str = None) -> Dict[str, np.ndarray]:
        """Generate embeddings for code using specified or all models."""
        if model_name:
            return {model_name: self._generate_single_embedding(code, model_name)}
        
        embeddings = {}
        for model in self.config['models']:
            try:
                embeddings[model] = self._generate_single_embedding(code, model)
            except Exception as e:
                logger.warning(f"Failed to generate embedding with {model}: {e}")
                continue
        
        return embeddings
    
    def _generate_single_embedding(self, code: str, model_name: str) -> np.ndarray:
        """Generate embedding using a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        # Tokenize code
        inputs = tokenizer(
            code,
            return_tensors='pt',
            max_length=self.config['max_length'],
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use [CLS] token embedding or mean pooling
            if hasattr(outputs, 'last_hidden_state'):
                # Mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                # Use pooler output if available
                embeddings = outputs.pooler_output
            
            return embeddings.cpu().numpy().flatten()
    
    def generate_batch_embeddings(self, code_list: List[str], model_name: str = None) -> Dict[str, np.ndarray]:
        """Generate embeddings for a batch of code samples."""
        if model_name:
            return {model_name: self._generate_batch_single_embedding(code_list, model_name)}
        
        embeddings = {}
        for model in self.config['models']:
            try:
                embeddings[model] = self._generate_batch_single_embedding(code_list, model)
            except Exception as e:
                logger.warning(f"Failed to generate batch embeddings with {model}: {e}")
                continue
        
        return embeddings
    
    def _generate_batch_single_embedding(self, code_list: List[str], model_name: str) -> np.ndarray:
        """Generate embeddings for a batch using a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        # Tokenize batch
        inputs = tokenizer(
            code_list,
            return_tensors='pt',
            max_length=self.config['max_length'],
            padding=True,
            truncation=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use [CLS] token embedding or mean pooling
            if hasattr(outputs, 'last_hidden_state'):
                # Mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                # Use pooler output if available
                embeddings = outputs.pooler_output
            
            return embeddings.cpu().numpy()
    
    def generate_multi_modal_embeddings(self, code: str) -> Dict[str, np.ndarray]:
        """Generate multi-modal embeddings combining different approaches."""
        embeddings = {}
        
        # Generate embeddings from different models
        model_embeddings = self.generate_embeddings(code)
        embeddings.update(model_embeddings)
        
        # Generate custom embeddings
        custom_embeddings = self._generate_custom_embeddings(code)
        embeddings.update(custom_embeddings)
        
        return embeddings
    
    def _generate_custom_embeddings(self, code: str) -> Dict[str, np.ndarray]:
        """Generate custom embeddings using domain-specific techniques."""
        embeddings = {}
        
        # Character-level embeddings
        embeddings['char_level'] = self._generate_character_embeddings(code)
        
        # Token-level embeddings
        embeddings['token_level'] = self._generate_token_embeddings(code)
        
        # Structural embeddings
        embeddings['structural'] = self._generate_structural_embeddings(code)
        
        return embeddings
    
    def _generate_character_embeddings(self, code: str) -> np.ndarray:
        """Generate character-level embeddings."""
        # Simple character frequency vector
        char_freq = np.zeros(256)  # ASCII characters
        for char in code:
            if ord(char) < 256:
                char_freq[ord(char)] += 1
        
        # Normalize
        if char_freq.sum() > 0:
            char_freq = char_freq / char_freq.sum()
        
        return char_freq
    
    def _generate_token_embeddings(self, code: str) -> np.ndarray:
        """Generate token-level embeddings."""
        import re
        
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', code)
        
        # Create token frequency vector
        token_freq = {}
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
        
        # Convert to fixed-size vector (top 1000 most common tokens)
        common_tokens = [
            'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while',
            'try', 'except', 'finally', 'with', 'as', 'return', 'yield', 'lambda',
            'and', 'or', 'not', 'in', 'is', 'None', 'True', 'False', 'self',
            'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'sorted',
            'list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool'
        ]
        
        token_vector = np.zeros(len(common_tokens))
        for i, token in enumerate(common_tokens):
            token_vector[i] = token_freq.get(token, 0)
        
        # Normalize
        if token_vector.sum() > 0:
            token_vector = token_vector / token_vector.sum()
        
        return token_vector
    
    def _generate_structural_embeddings(self, code: str) -> np.ndarray:
        """Generate structural embeddings based on code structure."""
        lines = code.split('\n')
        
        # Structural features
        features = [
            len(lines),  # Number of lines
            len([line for line in lines if line.strip()]),  # Non-empty lines
            len(code),  # Total characters
            len(code.split()),  # Total words
            code.count('def '),  # Function definitions
            code.count('class '),  # Class definitions
            code.count('import '),  # Import statements
            code.count('if '),  # If statements
            code.count('for '),  # For loops
            code.count('while '),  # While loops
            code.count('try:'),  # Try blocks
            code.count('except'),  # Except blocks
            code.count('return '),  # Return statements
            code.count('yield '),  # Yield statements
            code.count('lambda '),  # Lambda functions
        ]
        
        return np.array(features, dtype=np.float32)
    
    def generate_attention_embeddings(self, code: str, model_name: str = 'codebert') -> Tuple[np.ndarray, np.ndarray]:
        """Generate embeddings with attention weights for explainability."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        # Tokenize code
        inputs = tokenizer(
            code,
            return_tensors='pt',
            max_length=self.config['max_length'],
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings with attention
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
            # Get embeddings
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                embeddings = outputs.pooler_output
            
            # Get attention weights (average across all layers and heads)
            attention_weights = torch.stack(outputs.attentions).mean(dim=(0, 1))  # Average over layers and heads
            attention_weights = attention_weights.mean(dim=0)  # Average over query positions
            
            return embeddings.cpu().numpy().flatten(), attention_weights.cpu().numpy()
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], filepath: str):
        """Save embeddings to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load embeddings from file."""
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        
        logger.info(f"Embeddings loaded from {filepath}")
        return embeddings
    
    def get_embedding_dimension(self, model_name: str) -> int:
        """Get embedding dimension for a specific model."""
        if model_name == 'codebert':
            return 768
        elif model_name == 'graphcodebert':
            return 768
        elif model_name == 'roberta':
            return 768
        elif model_name == 'char_level':
            return 256
        elif model_name == 'token_level':
            return 50  # Based on common_tokens length
        elif model_name == 'structural':
            return 15  # Based on features length
        else:
            return self.config['embedding_dim']
    
    def create_embedding_matrix(self, code_list: List[str], model_name: str = 'codebert') -> np.ndarray:
        """Create embedding matrix for a list of code samples."""
        embeddings = self.generate_batch_embeddings(code_list, model_name)
        return embeddings[model_name]
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity matrix between embeddings."""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings)
    
    def reduce_dimensionality(self, embeddings: np.ndarray, method: str = 'pca', n_components: int = 50) -> np.ndarray:
        """Reduce dimensionality of embeddings."""
        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            return pca.fit_transform(embeddings)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=n_components, random_state=42)
            return tsne.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

class EmbeddingEnsemble:
    """Ensemble of different embedding methods."""
    
    def __init__(self, embedding_generator: CodeEmbeddingGenerator):
        self.embedding_generator = embedding_generator
    
    def generate_ensemble_embeddings(self, code: str) -> np.ndarray:
        """Generate ensemble embeddings by combining multiple methods."""
        # Get embeddings from different models
        model_embeddings = self.embedding_generator.generate_embeddings(code)
        
        # Get custom embeddings
        custom_embeddings = self.embedding_generator._generate_custom_embeddings(code)
        
        # Combine all embeddings
        all_embeddings = {**model_embeddings, **custom_embeddings}
        
        # Concatenate embeddings
        ensemble_embedding = np.concatenate(list(all_embeddings.values()))
        
        return ensemble_embedding
    
    def generate_weighted_embeddings(self, code: str, weights: Dict[str, float] = None) -> np.ndarray:
        """Generate weighted ensemble embeddings."""
        if weights is None:
            weights = {
                'codebert': 0.3,
                'graphcodebert': 0.3,
                'char_level': 0.1,
                'token_level': 0.1,
                'structural': 0.2
            }
        
        # Get all embeddings
        model_embeddings = self.embedding_generator.generate_embeddings(code)
        custom_embeddings = self.embedding_generator._generate_custom_embeddings(code)
        all_embeddings = {**model_embeddings, **custom_embeddings}
        
        # Apply weights and combine
        weighted_embeddings = []
        for name, embedding in all_embeddings.items():
            weight = weights.get(name, 0.0)
            weighted_embeddings.append(weight * embedding)
        
        return np.concatenate(weighted_embeddings)
