"""
Advanced transformer-based model for AI code detection.
Uses CodeBERT, GraphCodeBERT, and custom transformer architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    RobertaTokenizer, RobertaModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

logger = logging.getLogger("ai_code_detector")

class CodeDataset(Dataset):
    """Custom dataset for code detection."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """Initialize dataset."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CodeDetectionModel(nn.Module):
    """Custom transformer model for code detection."""
    
    def __init__(self, model_name: str, num_classes: int = 2, dropout_rate: float = 0.1):
        """Initialize model."""
        super(CodeDetectionModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained model
        if model_name == 'codebert':
            self.encoder = AutoModel.from_pretrained('microsoft/codebert-base')
            self.hidden_size = 768
        elif model_name == 'graphcodebert':
            self.encoder = AutoModel.from_pretrained('microsoft/graphcodebert-base')
            self.hidden_size = 768
        elif model_name == 'roberta':
            self.encoder = RobertaModel.from_pretrained('roberta-base')
            self.hidden_size = 768
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
        # Attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
    
    def forward(self, input_ids, attention_mask, return_attention=False):
        """Forward pass."""
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply attention pooling
        pooled_output, attention_weights = self.attention_pooling(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Apply layer normalization
        pooled_output = self.layer_norm(pooled_output)
        
        # Global average pooling
        pooled_output = pooled_output.mean(dim=1)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits

class TransformerTrainer:
    """Trainer for transformer models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize trainer."""
        self.config = config or {
            'model_name': 'codebert',
            'max_length': 512,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'epochs': 10,
            'warmup_steps': 100,
            'weight_decay': 0.01,
            'gradient_clip_norm': 1.0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Safe default for device even if key is missing
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        self.tokenizer = None
        self.training_history = []
        
        # Initialize tokenizer
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load appropriate tokenizer."""
        model_name = self.config['model_name']
        
        if model_name == 'codebert':
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        elif model_name == 'graphcodebert':
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
        elif model_name == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        logger.info(f"Loaded tokenizer for {model_name}")
    
    def create_model(self, num_classes: int = 2) -> CodeDetectionModel:
        """Create model instance."""
        self.model = CodeDetectionModel(
            model_name=self.config['model_name'],
            num_classes=num_classes,
            dropout_rate=self.config.get('dropout_rate', 0.1)
        )
        
        self.model.to(self.device)
        logger.info(f"Created model: {self.config['model_name']}")
        
        return self.model
    
    def train(self, X_train: List[str], y_train: List[int], 
              X_val: List[str] = None, y_val: List[int] = None) -> Dict[str, Any]:
        """Train the model."""
        if self.model is None:
            self.create_model()
        
        # Create datasets
        train_dataset = CodeDataset(
            X_train, y_train, self.tokenizer, self.config['max_length']
        )
        
        val_dataset = None
        if X_val is not None and y_val is not None:
            val_dataset = CodeDataset(
                X_val, y_val, self.tokenizer, self.config['max_length']
            )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False
            )
        
        # Setup training
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        total_steps = len(train_loader) * self.config['epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_score = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler, criterion)
            
            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                **val_metrics
            }
            
            self.training_history.append(epoch_metrics)
            
            logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            if val_metrics:
                logger.info(f"Val Loss: {val_metrics.get('val_loss', 0):.4f}, Val Acc: {val_metrics.get('val_accuracy', 0):.4f}")
            
            # Early stopping
            if val_metrics and 'val_f1' in val_metrics:
                if val_metrics['val_f1'] > best_val_score:
                    best_val_score = val_metrics['val_f1']
                    patience_counter = 0
                    # Save best model
                    self._save_checkpoint(epoch, val_metrics['val_f1'])
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
        
        return {
            'training_history': self.training_history,
            'best_val_score': best_val_score
        }
    
    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler, criterion) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['gradient_clip_norm']
            )
            
            optimizer.step()
            scheduler.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
    
    def _validate_epoch(self, val_loader: DataLoader, criterion) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_f1': f1,
            'val_precision': precision,
            'val_recall': recall
        }
    
    def predict(self, X: List[str]) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        predictions = []
        
        # Create dataset
        dataset = CodeDataset(X, [0] * len(X), self.tokenizer, self.config['max_length'])
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                batch_predictions = torch.argmax(logits, dim=1)
                predictions.extend(batch_predictions.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        probabilities = []
        
        # Create dataset
        dataset = CodeDataset(X, [0] * len(X), self.tokenizer, self.config['max_length'])
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                batch_probs = torch.softmax(logits, dim=1)
                probabilities.extend(batch_probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def get_attention_weights(self, X: List[str]) -> List[np.ndarray]:
        """Get attention weights for explainability."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        attention_weights = []
        
        # Create dataset
        dataset = CodeDataset(X, [0] * len(X), self.tokenizer, self.config['max_length'])
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Batch size 1 for attention
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits, attn_weights = self.model(input_ids, attention_mask, return_attention=True)
                attention_weights.append(attn_weights.cpu().numpy())
        
        return attention_weights
    
    def _save_checkpoint(self, epoch: int, score: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'score': score,
            'config': self.config
        }
        
        checkpoint_path = Path("models/transformer/checkpoints")
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path / f"checkpoint_epoch_{epoch}.pt")
        logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def save_model(self, save_path: str = "models/transformer"):
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), save_path / "model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save training history
        with open(save_path / "training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str = "models/transformer"):
        """Load trained model."""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model path not found: {load_path}")
        
        # Load config
        with open(load_path / "config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Create and load model
        self.create_model()
        self.model.load_state_dict(torch.load(load_path / "model.pt", map_location=self.device))
        
        # Load training history
        if (load_path / "training_history.json").exists():
            with open(load_path / "training_history.json", 'r') as f:
                self.training_history = json.load(f)
        
        logger.info(f"Model loaded from {load_path}")

class MultiModelEnsemble:
    """Ensemble of multiple transformer models."""
    
    def __init__(self, model_configs: List[Dict[str, Any]]):
        """Initialize with multiple model configurations."""
        self.model_configs = model_configs
        self.models = {}
        self.trainers = {}
        
        # Initialize trainers
        for config in model_configs:
            model_name = config['model_name']
            self.trainers[model_name] = TransformerTrainer(config)
    
    def train_all_models(self, X_train: List[str], y_train: List[int], 
                        X_val: List[str] = None, y_val: List[int] = None) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        results = {}
        
        for model_name, trainer in self.trainers.items():
            logger.info(f"Training {model_name}")
            
            try:
                result = trainer.train(X_train, y_train, X_val, y_val)
                results[model_name] = result
                self.models[model_name] = trainer.model
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def predict_ensemble(self, X: List[str], method: str = 'voting') -> np.ndarray:
        """Make ensemble predictions."""
        if not self.models:
            raise ValueError("No models trained yet")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from all models
        for model_name, trainer in self.trainers.items():
            if model_name in self.models:
                predictions[model_name] = trainer.predict(X)
                probabilities[model_name] = trainer.predict_proba(X)
        
        if method == 'voting':
            # Majority voting
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=pred_array
            )
            return ensemble_pred
        
        elif method == 'averaging':
            # Average probabilities
            prob_array = np.array(list(probabilities.values()))
            avg_probs = np.mean(prob_array, axis=0)
            return np.argmax(avg_probs, axis=1)
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def get_ensemble_attention(self, X: List[str]) -> Dict[str, List[np.ndarray]]:
        """Get attention weights from all models."""
        attention_weights = {}
        
        for model_name, trainer in self.trainers.items():
            if model_name in self.models:
                attention_weights[model_name] = trainer.get_attention_weights(X)
        
        return attention_weights
