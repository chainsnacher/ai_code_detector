"""
Baseline machine learning models for AI code detection.
Includes Random Forest, SVM, Logistic Regression, and ensemble methods.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time

logger = logging.getLogger("ai_code_detector")


def _sanitize_array(X):
    """Top-level sanitizer function for numeric arrays.

    This replaces the lambda used previously so the FunctionTransformer is pickleable.
    """
    import numpy as _np
    try:
        X = _np.nan_to_num(_np.clip(X, -1e6, 1e6), nan=0.0, posinf=1e6, neginf=-1e6)
    except Exception:
        # If input isn't numeric or shape is unexpected, try converting
        X = _np.asarray(X, dtype=float)
        X = _np.nan_to_num(_np.clip(X, -1e6, 1e6), nan=0.0, posinf=1e6, neginf=-1e6)

    return X

class BaselineModelTrainer:
    """Trains and manages baseline machine learning models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize baseline model trainer."""
        self.config = config or {}
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_history = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced_subsample',
                'random_state': 42,
                'n_jobs': -1
            },
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'class_weight': 'balanced',
                'random_state': 42,
                'probability': True
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
                'solver': 'liblinear'
            },
            'gradient_boosting': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train all baseline models."""
        logger.info("Starting baseline model training")
        
        results = {}
        
        for model_name in self.model_configs.keys():
            logger.info(f"Training {model_name}")
            start_time = time.time()
            
            try:
                model_result = self._train_single_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                results[model_name] = model_result
                
                training_time = time.time() - start_time
                logger.info(f"{model_name} training completed in {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def _train_single_model(self, model_name: str, X_train: np.ndarray, 
                           y_train: np.ndarray, X_val: np.ndarray = None, 
                           y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train a single model with hyperparameter optimization."""

        
        
        # Create pipeline with sanitization and scaling
        scaler = StandardScaler()
        model = self._create_model(model_name)

        sanitizer = FunctionTransformer(_sanitize_array, validate=False)

        pipeline = Pipeline([
            ('sanitize', sanitizer),
            ('scaler', scaler),
            ('classifier', model)
        ])
        
        # Hyperparameter optimization
        param_grid = self._get_param_grid(model_name)
        
        if param_grid:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=0, error_score='raise'
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
        else:
            best_model = pipeline
            best_model.fit(X_train, y_train)
            best_params = self.model_configs[model_name]
            best_score = None
        
        # Store model and scaler
        self.models[model_name] = best_model
        self.scalers[model_name] = scaler
        
        # Evaluate model
        train_score = best_model.score(X_train, y_train)
        val_score = best_model.score(X_val, y_val) if X_val is not None else None
        
        # Get feature importance
        feature_importance = self._get_feature_importance(best_model, model_name)
        self.feature_importance[model_name] = feature_importance
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
        
        result = {
            'model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'train_score': train_score,
            'val_score': val_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
        
        return result
    
    def _create_model(self, model_name: str):
        """Create model instance based on name."""
        if model_name == 'random_forest':
            return RandomForestClassifier(**self.model_configs[model_name])
        elif model_name == 'svm':
            return SVC(**self.model_configs[model_name])
        elif model_name == 'logistic_regression':
            return LogisticRegression(**self.model_configs[model_name])
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(**self.model_configs[model_name])
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _get_param_grid(self, model_name: str) -> Dict[str, List]:
        """Get parameter grid for hyperparameter optimization."""
        param_grids = {
            'random_forest': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5, 10]
            },
            'svm': {
                'classifier__C': [0.1, 1, 10],
                'classifier__gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'logistic_regression': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear', 'lbfgs']
            },
            'gradient_boosting': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6, 10],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        return param_grids.get(model_name, {})
    
    def _get_feature_importance(self, model, model_name: str) -> np.ndarray:
        """Get feature importance from trained model."""
        try:
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                return model.named_steps['classifier'].feature_importances_
            elif hasattr(model.named_steps['classifier'], 'coef_'):
                return np.abs(model.named_steps['classifier'].coef_[0])
            else:
                return np.array([])
        except Exception as e:
            logger.warning(f"Could not extract feature importance for {model_name}: {e}")
            return np.array([])
    
    def create_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
        """Create ensemble model from trained individual models."""
        if not self.models:
            raise ValueError("No models trained yet. Train models first.")
        
        # Create voting classifier
        estimators = []
        for model_name, model in self.models.items():
            estimators.append((model_name, model))
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use predicted probabilities
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Store ensemble
        self.models['ensemble'] = ensemble
        
        return ensemble
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """Make predictions using specified model or best model."""
        if model_name is None:
            model_name = self._get_best_model()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict(X)
    
    def predict_proba(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """Get prediction probabilities."""
        if model_name is None:
            model_name = self._get_best_model()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models[model_name].predict_proba(X)
    
    def _get_best_model(self) -> str:
        """Get the best performing model based on validation score."""
        if not self.training_history:
            return list(self.models.keys())[0]
        
        best_model = max(
            self.training_history.items(),
            key=lambda x: x[1].get('val_score', 0)
        )[0]
        
        return best_model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str = None) -> Dict[str, Any]:
        """Evaluate model performance on test set."""
        if model_name is None:
            model_name = self._get_best_model()
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = model.score(X_test, y_test)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        evaluation = {
            'model_name': model_name,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
        }
        
        return evaluation
    
    def save_models(self, save_dir: str = "models/baseline"):
        """Save all trained models."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_file = save_path / f"{model_name}_model.pkl"
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} model to {model_file}")
        
        # Save feature importance
        importance_file = save_path / "feature_importance.pkl"
        joblib.dump(self.feature_importance, importance_file)
        logger.info(f"Saved feature importance to {importance_file}")
    
    def load_models(self, load_dir: str = "models/baseline"):
        """Load pre-trained models."""
        load_path = Path(load_dir)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model directory not found: {load_dir}")
        
        # Load models
        for model_file in load_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace("_model", "")
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded {model_name} model from {model_file}")
        
        # Load feature importance
        importance_file = load_path / "feature_importance.pkl"
        if importance_file.exists():
            self.feature_importance = joblib.load(importance_file)
            logger.info(f"Loaded feature importance from {importance_file}")
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models."""
        if not self.training_history:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, history in self.training_history.items():
            summary_data.append({
                'Model': model_name,
                'Train Score': history.get('train_score', 0),
                'Val Score': history.get('val_score', 0),
                'CV Mean': history.get('cv_mean', 0),
                'CV Std': history.get('cv_std', 0),
                'Best Params': str(history.get('best_params', {}))
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_feature_importance(self, model_name: str, top_n: int = 20) -> None:
        """Plot feature importance for a model."""
        import matplotlib.pyplot as plt
        
        if model_name not in self.feature_importance:
            logger.warning(f"No feature importance data for {model_name}")
            return
        
        importance = self.feature_importance[model_name]
        if len(importance) == 0:
            logger.warning(f"Empty feature importance for {model_name}")
            return
        
        # Get top N features
        top_indices = np.argsort(importance)[-top_n:]
        top_importance = importance[top_indices]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_importance)), top_importance)
        plt.yticks(range(len(top_importance)), [f"Feature_{i}" for i in top_indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.tight_layout()
        plt.show()

class AdvancedEnsemble:
    """Advanced ensemble methods for code detection."""
    
    def __init__(self, base_models: Dict[str, Any]):
        """Initialize with base models."""
        self.base_models = base_models
        self.meta_model = None
        self.meta_features = None
    
    def create_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Create meta-features from base model predictions."""
        meta_features = []
        
        for model_name, model in self.base_models.items():
            # Get predictions and probabilities
            predictions = model.predict(X)
            probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            # Add prediction as meta-feature
            meta_features.append(predictions.reshape(-1, 1))
            
            # Add probabilities as meta-features
            if probabilities is not None:
                meta_features.append(probabilities)
            
            # Add confidence scores
            if probabilities is not None:
                confidence = np.max(probabilities, axis=1)
                meta_features.append(confidence.reshape(-1, 1))
        
        return np.hstack(meta_features)
    
    def train_meta_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train meta-model on meta-features."""
        # Create meta-features
        self.meta_features = self.create_meta_features(X_train)
        
        # Train meta-model (Random Forest)
        self.meta_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        self.meta_model.fit(self.meta_features, y_train)
        logger.info("Meta-model trained successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using meta-model."""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained yet")
        
        meta_features = self.create_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from meta-model."""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained yet")
        
        meta_features = self.create_meta_features(X)
        return self.meta_model.predict_proba(meta_features)
