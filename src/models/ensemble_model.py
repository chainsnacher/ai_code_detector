"""
Advanced ensemble model for AI code detection.
Combines multiple models with meta-learning and confidence weighting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import json
import time
from collections import defaultdict

logger = logging.getLogger("ai_code_detector")

class MetaFeatureGenerator:
    """Generates meta-features from base model predictions."""
    
    def __init__(self):
        self.feature_names = []
    
    def generate_meta_features(self, predictions: Dict[str, np.ndarray], 
                             probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate meta-features from base model outputs."""
        meta_features = []
        feature_names = []
        # Determine sample count robustly
        n_samples = None
        for arr in predictions.values():
            if arr is not None:
                n_samples = len(arr)
                break
        if n_samples is None:
            for arr in probabilities.values():
                if arr is not None:
                    n_samples = arr.shape[0]
                    break
        if n_samples is None:
            n_samples = 1
        
        # Basic predictions
        for model_name, preds in predictions.items():
            if preds is None:
                continue
            meta_features.append(preds.reshape(-1, 1))
            feature_names.append(f"{model_name}_prediction")
        
        # Probabilities
        for model_name, probs in probabilities.items():
            if probs is None:
                continue
            if probs.ndim == 1:
                # treat as positive class probability
                meta_features.append(probs.reshape(-1, 1))
                feature_names.append(f"{model_name}_prob")
            elif probs.shape[1] == 2:  # Binary classification
                meta_features.append(probs[:, 1].reshape(-1, 1))  # Positive class probability
                feature_names.append(f"{model_name}_prob_positive")
            else:
                # Multi-class: add all probabilities
                for i in range(probs.shape[1]):
                    meta_features.append(probs[:, i].reshape(-1, 1))
                    feature_names.append(f"{model_name}_prob_class_{i}")
        
        # Confidence scores
        for model_name, probs in probabilities.items():
            if probs is None:
                continue
            confidence = np.max(probs, axis=1)
            meta_features.append(confidence.reshape(-1, 1))
            feature_names.append(f"{model_name}_confidence")
        
        # Model agreement features
        if len(predictions) > 1:
            pred_array = np.array(list(predictions.values()))
            
            # Agreement ratio
            agreement_ratio = self._calculate_agreement_ratio(pred_array)
            meta_features.append(agreement_ratio.reshape(-1, 1))
            feature_names.append("model_agreement_ratio")
            
            # Entropy of predictions
            prediction_entropy = self._calculate_prediction_entropy(pred_array)
            meta_features.append(prediction_entropy.reshape(-1, 1))
            feature_names.append("prediction_entropy")
            
            # Variance of probabilities
            prob_list = [p for p in probabilities.values() if p is not None]
            if prob_list:
                prob_array = np.array(prob_list)
            else:
                prob_array = None
            if prob_array is not None:
                prob_variance = np.var(prob_array, axis=0).mean(axis=1)
                meta_features.append(prob_variance.reshape(-1, 1))
                feature_names.append("probability_variance")
        
        # Confidence statistics
        all_confidences = []
        for probs in probabilities.values():
            if probs is None:
                continue
            all_confidences.append(np.max(probs, axis=1))
        
        if all_confidences:
            conf_array = np.array(all_confidences)
            
            # Mean confidence
            mean_confidence = np.mean(conf_array, axis=0)
            meta_features.append(mean_confidence.reshape(-1, 1))
            feature_names.append("mean_confidence")
            
            # Max confidence
            max_confidence = np.max(conf_array, axis=0)
            meta_features.append(max_confidence.reshape(-1, 1))
            feature_names.append("max_confidence")
            
            # Min confidence
            min_confidence = np.min(conf_array, axis=0)
            meta_features.append(min_confidence.reshape(-1, 1))
            feature_names.append("min_confidence")
            
            # Confidence range
            confidence_range = max_confidence - min_confidence
            meta_features.append(confidence_range.reshape(-1, 1))
            feature_names.append("confidence_range")
        
        self.feature_names = feature_names
        if not meta_features:
            # Return a placeholder zero feature to avoid downstream shape errors
            return np.zeros((n_samples, 1), dtype=float)
        return np.hstack(meta_features)
    
    def _calculate_agreement_ratio(self, pred_array: np.ndarray) -> np.ndarray:
        """Calculate agreement ratio between models."""
        n_models = pred_array.shape[0]
        agreement_ratios = []
        
        for i in range(pred_array.shape[1]):
            predictions = pred_array[:, i]
            most_common = np.bincount(predictions).argmax()
            agreement_count = np.sum(predictions == most_common)
            agreement_ratio = agreement_count / n_models
            agreement_ratios.append(agreement_ratio)
        
        return np.array(agreement_ratios)
    
    def _calculate_prediction_entropy(self, pred_array: np.ndarray) -> np.ndarray:
        """Calculate entropy of predictions."""
        entropies = []
        
        for i in range(pred_array.shape[1]):
            predictions = pred_array[:, i]
            unique, counts = np.unique(predictions, return_counts=True)
            probabilities = counts / len(predictions)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropies.append(entropy)
        
        return np.array(entropies)

class ConfidenceWeightedEnsemble:
    """Ensemble with confidence-based weighting."""
    
    def __init__(self, base_models: Dict[str, Any], confidence_threshold: float = 0.7):
        """Initialize confidence-weighted ensemble."""
        self.base_models = base_models
        self.confidence_threshold = confidence_threshold
        self.model_weights = {}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the ensemble by learning model weights."""
        # Get predictions and probabilities from all models
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.base_models.items():
            try:
                preds = model.predict(X)
                probs = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                
                predictions[model_name] = preds
                probabilities[model_name] = probs
                
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_name}: {e}")
                continue
        
        # Calculate model weights based on performance
        self._calculate_model_weights(predictions, probabilities, y)
        self.is_fitted = True
    
    def _calculate_model_weights(self, predictions: Dict[str, np.ndarray], 
                               probabilities: Dict[str, np.ndarray], y: np.ndarray):
        """Calculate weights for each model based on performance."""
        weights = {}
        
        for model_name, preds in predictions.items():
            # Calculate accuracy
            accuracy = accuracy_score(y, preds)
            
            # Calculate confidence-weighted accuracy
            if model_name in probabilities:
                probs = probabilities[model_name]
                confidence = np.max(probs, axis=1)
                
                # Weight accuracy by confidence
                confidence_weighted_accuracy = np.sum(
                    (preds == y) * confidence
                ) / np.sum(confidence)
                
                # Combine accuracy and confidence-weighted accuracy
                weight = 0.7 * accuracy + 0.3 * confidence_weighted_accuracy
            else:
                weight = accuracy
            
            weights[model_name] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.model_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights if all models failed
            self.model_weights = {k: 1.0 / len(weights) for k in weights.keys()}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using confidence-weighted voting."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from all models
        for model_name, model in self.base_models.items():
            try:
                preds = model.predict(X)
                probs = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                
                predictions[model_name] = preds
                probabilities[model_name] = probs
                
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_name}: {e}")
                continue
        
        # Weighted voting
        weighted_predictions = np.zeros(len(X))
        
        for model_name, preds in predictions.items():
            weight = self.model_weights.get(model_name, 0)
            weighted_predictions += weight * preds
        
        # Apply confidence threshold
        if probabilities:
            # Calculate average confidence
            all_confidences = []
            for probs in probabilities.values():
                if probs is not None:
                    all_confidences.append(np.max(probs, axis=1))
            
            if all_confidences:
                avg_confidence = np.mean(all_confidences, axis=0)
                
                # For low confidence predictions, use majority voting
                low_confidence_mask = avg_confidence < self.confidence_threshold
                
                if np.any(low_confidence_mask):
                    # Majority voting for low confidence cases
                    pred_array = np.array(list(predictions.values()))
                    majority_preds = np.apply_along_axis(
                        lambda x: np.bincount(x).argmax(), axis=0, arr=pred_array
                    )
                    
                    weighted_predictions[low_confidence_mask] = majority_preds[low_confidence_mask]
        
        return np.round(weighted_predictions).astype(int)

class AdvancedEnsembleDetector:
    """Advanced ensemble detector with multiple strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize advanced ensemble detector."""
        # Merge user config with sensible defaults to avoid KeyError on missing keys
        defaults = {
            'meta_classifier': 'random_forest',
            'confidence_threshold': 0.7,
            'use_confidence_weighting': True,
            'use_meta_features': True,
            'voting_strategy': 'soft'
        }
        self.config = {**defaults, **(config or {})}
        
        self.base_models = {}
        self.meta_classifier = None
        self.meta_feature_generator = MetaFeatureGenerator()
        self.confidence_ensemble = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Performance tracking
        self.model_performance = {}
        self.ensemble_performance = {}
    
    def add_base_model(self, name: str, model: Any):
        """Add a base model to the ensemble."""
        self.base_models[name] = model
        logger.info(f"Added base model: {name}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Fit the ensemble with all strategies."""
        if not self.base_models:
            logger.warning("No base models added; skipping ensemble fit")
            return
        
        logger.info("Fitting advanced ensemble detector")
        
        # 1. Evaluate individual models
        self._evaluate_base_models(X, y, X_val, y_val)
        
        # 2. Fit confidence-weighted ensemble
        if self.config['use_confidence_weighting']:
            self.confidence_ensemble = ConfidenceWeightedEnsemble(
                self.base_models, 
                self.config['confidence_threshold']
            )
            self.confidence_ensemble.fit(X, y)
        
        # 3. Fit meta-classifier
        if self.config['use_meta_features']:
            self._fit_meta_classifier(X, y, X_val, y_val)
        
        self.is_fitted = True
        logger.info("Advanced ensemble detector fitted successfully")
    
    def _evaluate_base_models(self, X: np.ndarray, y: np.ndarray, 
                            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Evaluate performance of base models."""
        logger.info("Evaluating base models")
        
        for model_name, model in self.base_models.items():
            try:
                # Training performance
                train_preds = model.predict(X)
                train_acc = accuracy_score(y, train_preds)
                train_f1 = f1_score(y, train_preds, average='weighted')
                
                # Validation performance
                val_acc = None
                val_f1 = None
                if X_val is not None and y_val is not None:
                    val_preds = model.predict(X_val)
                    val_acc = accuracy_score(y_val, val_preds)
                    val_f1 = f1_score(y_val, val_preds, average='weighted')
                
                self.model_performance[model_name] = {
                    'train_accuracy': train_acc,
                    'train_f1': train_f1,
                    'val_accuracy': val_acc,
                    'val_f1': val_f1
                }
                # Format validation accuracy safely. Use explicit None check so 0.0 is not treated as missing.
                if val_acc is not None:
                    val_acc_str = f"{val_acc:.4f}"
                else:
                    val_acc_str = "N/A"

                logger.info(f"{model_name} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc_str}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                self.model_performance[model_name] = {'error': str(e)}
    
    def _fit_meta_classifier(self, X: np.ndarray, y: np.ndarray, 
                           X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Fit meta-classifier on meta-features."""
        logger.info("Fitting meta-classifier")
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X)
        
        # Scale features
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        
        # Create meta-classifier
        if self.config['meta_classifier'] == 'random_forest':
            self.meta_classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.config['meta_classifier'] == 'logistic_regression':
            self.meta_classifier = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown meta-classifier: {self.config['meta_classifier']}")
        
        # Train meta-classifier
        self.meta_classifier.fit(meta_features_scaled, y)
        
        # Evaluate meta-classifier
        meta_preds = self.meta_classifier.predict(meta_features_scaled)
        meta_acc = accuracy_score(y, meta_preds)
        meta_f1 = f1_score(y, meta_preds, average='weighted')
        
        self.ensemble_performance['meta_classifier'] = {
            'train_accuracy': meta_acc,
            'train_f1': meta_f1
        }
        
        logger.info(f"Meta-classifier - Train Acc: {meta_acc:.4f}, Train F1: {meta_f1:.4f}")
    
    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base model predictions."""
        predictions = {}
        probabilities = {}
        
        # Get predictions from all base models
        for model_name, model in self.base_models.items():
            try:
                preds = model.predict(X)
                probs = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                
                predictions[model_name] = preds
                probabilities[model_name] = probs
                
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_name}: {e}")
                continue
        
        # Generate meta-features
        meta_features = self.meta_feature_generator.generate_meta_features(
            predictions, probabilities
        )
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        # Get base model predictions
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.base_models.items():
            try:
                preds = model.predict(X)
                probs = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
                
                predictions[model_name] = preds
                probabilities[model_name] = probs
                
            except Exception as e:
                logger.warning(f"Error getting predictions from {model_name}: {e}")
                continue
        
        # Strategy 1: Meta-classifier (if available)
        if self.meta_classifier is not None:
            meta_features = self.meta_feature_generator.generate_meta_features(
                predictions, probabilities
            )
            meta_features_scaled = self.scaler.transform(meta_features)
            # Prefer probability-driven decision when available
            if hasattr(self.meta_classifier, 'predict_proba'):
                meta_probs = self.meta_classifier.predict_proba(meta_features_scaled)
                # Use positive class probability for binary case
                if meta_probs.ndim == 2 and meta_probs.shape[1] == 2:
                    meta_preds = (meta_probs[:, 1] >= 0.5).astype(int)
                else:
                    meta_preds = self.meta_classifier.predict(meta_features_scaled)
            else:
                meta_preds = self.meta_classifier.predict(meta_features_scaled)
                meta_probs = None
        else:
            meta_preds = None
            meta_probs = None
        
        # Strategy 2: Confidence-weighted ensemble
        if self.confidence_ensemble is not None:
            conf_preds = self.confidence_ensemble.predict(X)
        else:
            conf_preds = None
        
        # Strategy 3: Simple voting
        if len(predictions) > 1:
            pred_array = np.array(list(predictions.values()))
            vote_preds = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=0, arr=pred_array
            )
        else:
            vote_preds = list(predictions.values())[0]
        
        # Combine strategies
        final_predictions = self._combine_predictions(
            meta_preds, conf_preds, vote_preds, meta_probs, probabilities
        )
        
        return final_predictions
    
    def _combine_predictions(self, meta_preds: np.ndarray, conf_preds: np.ndarray, 
                           vote_preds: np.ndarray, meta_probs: np.ndarray, 
                           base_probs: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions from different strategies."""
        strategies = []
        weights = []
        
        # Meta-classifier
        if meta_preds is not None:
            strategies.append(meta_preds)
            weights.append(0.4)  # High weight for meta-classifier
        
        # Confidence-weighted
        if conf_preds is not None:
            strategies.append(conf_preds)
            weights.append(0.3)
        
        # Simple voting
        if vote_preds is not None:
            strategies.append(vote_preds)
            weights.append(0.3)
        
        if not strategies:
            raise ValueError("No prediction strategies available")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted combination
        final_preds = np.zeros(len(strategies[0]))
        for strategy, weight in zip(strategies, weights):
            final_preds += weight * strategy
        
        return np.round(final_preds).astype(int)
    
    def predict_single(self, code: str) -> Dict[str, Any]:
        """Make prediction for a single code sample."""
        # This would need to be adapted based on your feature extraction pipeline
        # For now, return a placeholder structure
        return {
            'prediction': 0,  # 0 for human, 1 for AI
            'confidence': 0.85,
            'explanation': 'Prediction based on ensemble of multiple models',
            'model_agreement': 0.9,
            'feature_importance': {}
        }
    
    def get_model_importance(self) -> Dict[str, float]:
        """Get importance scores for each base model."""
        if not self.model_performance:
            return {}
        
        # Calculate importance based on performance
        importance = {}
        for model_name, perf in self.model_performance.items():
            if 'error' not in perf:
                # Use validation F1 if available, otherwise training F1
                score = perf.get('val_f1', perf.get('train_f1', 0))
                importance[model_name] = score
        
        # Normalize to sum to 1
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def save_ensemble(self, save_path: str = "models/ensemble"):
        """Save the ensemble model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(save_path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save meta-classifier
        if self.meta_classifier is not None:
            joblib.dump(self.meta_classifier, save_path / "meta_classifier.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, save_path / "scaler.pkl")
        
        # Save meta-feature generator
        joblib.dump(self.meta_feature_generator, save_path / "meta_feature_generator.pkl")
        
        # Save performance metrics
        with open(save_path / "performance.json", 'w') as f:
            json.dump({
                'model_performance': self.model_performance,
                'ensemble_performance': self.ensemble_performance
            }, f, indent=2)
        
        logger.info(f"Ensemble saved to {save_path}")
    
    def load_ensemble(self, load_path: str = "models/ensemble"):
        """Load the ensemble model."""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Ensemble path not found: {load_path}")
        
        # Load configuration
        with open(load_path / "config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load meta-classifier
        meta_classifier_path = load_path / "meta_classifier.pkl"
        if meta_classifier_path.exists():
            self.meta_classifier = joblib.load(meta_classifier_path)
        
        # Load scaler
        self.scaler = joblib.load(load_path / "scaler.pkl")
        
        # Load meta-feature generator
        self.meta_feature_generator = joblib.load(load_path / "meta_feature_generator.pkl")
        
        # Load performance metrics
        performance_path = load_path / "performance.json"
        if performance_path.exists():
            with open(performance_path, 'r') as f:
                perf_data = json.load(f)
                self.model_performance = perf_data.get('model_performance', {})
                self.ensemble_performance = perf_data.get('ensemble_performance', {})
        
        self.is_fitted = True
        logger.info(f"Ensemble loaded from {load_path}")
