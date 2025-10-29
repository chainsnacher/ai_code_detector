"""
Comprehensive evaluation metrics for AI code detection.
Includes custom metrics, robustness evaluation, and explainability measures.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger("ai_code_detector")

class AdvancedMetrics:
    """Advanced evaluation metrics for code detection."""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        metrics['precision_human'] = precision_per_class[0] if len(precision_per_class) > 0 else 0
        metrics['precision_ai'] = precision_per_class[1] if len(precision_per_class) > 1 else 0
        metrics['recall_human'] = recall_per_class[0] if len(recall_per_class) > 0 else 0
        metrics['recall_ai'] = recall_per_class[1] if len(recall_per_class) > 1 else 0
        metrics['f1_human'] = f1_per_class[0] if len(f1_per_class) > 0 else 0
        metrics['f1_ai'] = f1_per_class[1] if len(f1_per_class) > 1 else 0
        
        # ROC AUC
        if y_proba is not None:
            if y_proba.ndim == 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            
            # Precision-Recall AUC
            if y_proba.ndim == 1:
                metrics['pr_auc'] = average_precision_score(y_true, y_proba)
            else:
                metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        metrics.update(self._calculate_confusion_matrix_metrics(cm))
        
        # Custom metrics
        metrics.update(self._calculate_custom_metrics(y_true, y_pred, y_proba))
        
        return metrics
    
    def _calculate_confusion_matrix_metrics(self, cm: np.ndarray) -> Dict[str, float]:
        """Calculate metrics from confusion matrix."""
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        }
        
        return metrics
    
    def _calculate_custom_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate custom metrics specific to code detection."""
        metrics = {}
        
        # Detection rate for AI code
        ai_mask = y_true == 1
        if np.any(ai_mask):
            ai_detection_rate = np.sum((y_pred == 1) & ai_mask) / np.sum(ai_mask)
            metrics['ai_detection_rate'] = ai_detection_rate
        
        # False alarm rate for human code
        human_mask = y_true == 0
        if np.any(human_mask):
            false_alarm_rate = np.sum((y_pred == 1) & human_mask) / np.sum(human_mask)
            metrics['false_alarm_rate'] = false_alarm_rate
        
        # Confidence calibration
        if y_proba is not None:
            metrics.update(self._calculate_calibration_metrics(y_true, y_pred, y_proba))
        
        # Robustness metrics
        metrics.update(self._calculate_robustness_metrics(y_true, y_pred))
        
        return metrics
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate calibration metrics."""
        metrics = {}
        
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]  # Use positive class probability
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_true, y_proba, n_bins=10)
        metrics['expected_calibration_error'] = ece
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(y_true, y_proba, n_bins=10)
        metrics['max_calibration_error'] = mce
        
        # Reliability diagram metrics
        reliability_metrics = self._calculate_reliability_metrics(y_true, y_proba, n_bins=10)
        metrics.update(reliability_metrics)
        
        return metrics
    
    def _calculate_ece(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _calculate_reliability_metrics(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                     n_bins: int = 10) -> Dict[str, float]:
        """Calculate reliability diagram metrics."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                bin_accuracies.append(y_true[in_bin].mean())
                bin_confidences.append(y_proba[in_bin].mean())
                bin_counts.append(prop_in_bin)
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        # Calculate reliability metrics
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # Reliability score (lower is better)
        reliability_score = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / np.sum(bin_counts)
        
        return {
            'reliability_score': reliability_score,
            'bin_accuracies': bin_accuracies.tolist(),
            'bin_confidences': bin_confidences.tolist(),
            'bin_counts': bin_counts.tolist()
        }
    
    def _calculate_robustness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate robustness metrics."""
        metrics = {}
        
        # Prediction consistency (how often predictions change with small perturbations)
        # This is a simplified version - in practice, you'd test with actual perturbations
        prediction_entropy = self._calculate_prediction_entropy(y_pred)
        metrics['prediction_entropy'] = prediction_entropy
        
        # Class balance sensitivity
        class_balance = np.bincount(y_true) / len(y_true)
        metrics['class_balance_human'] = class_balance[0] if len(class_balance) > 0 else 0
        metrics['class_balance_ai'] = class_balance[1] if len(class_balance) > 1 else 0
        
        return metrics
    
    def _calculate_prediction_entropy(self, y_pred: np.ndarray) -> float:
        """Calculate entropy of predictions."""
        unique, counts = np.unique(y_pred, return_counts=True)
        probabilities = counts / len(y_pred)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5, scoring: str = 'f1_weighted') -> Dict[str, Any]:
        """Perform cross-validation with comprehensive metrics."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Basic cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        # Detailed cross-validation
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            fold_metric = self.calculate_comprehensive_metrics(y_val, y_pred, y_proba)
            fold_metric['fold'] = fold
            fold_metrics.append(fold_metric)
        
        # Aggregate metrics across folds
        metrics_df = pd.DataFrame(fold_metrics)
        aggregated_metrics = {}
        
        for metric in metrics_df.columns:
            if metric != 'fold':
                aggregated_metrics[f'{metric}_mean'] = metrics_df[metric].mean()
                aggregated_metrics[f'{metric}_std'] = metrics_df[metric].std()
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'fold_metrics': fold_metrics,
            'aggregated_metrics': aggregated_metrics
        }
    
    def plot_metrics(self, metrics: Dict[str, float], save_path: str = None):
        """Plot evaluation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Basic metrics bar plot
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        basic_values = [metrics.get(m, 0) for m in basic_metrics]
        
        axes[0, 0].bar(basic_metrics, basic_values, color=['blue', 'green', 'orange', 'red'])
        axes[0, 0].set_title('Basic Classification Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        # 2. Per-class metrics
        per_class_metrics = ['precision_human', 'precision_ai', 'recall_human', 'recall_ai']
        per_class_values = [metrics.get(m, 0) for m in per_class_metrics]
        
        x_pos = np.arange(len(per_class_metrics))
        axes[0, 1].bar(x_pos, per_class_values, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])
        axes[0, 1].set_title('Per-Class Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(per_class_metrics, rotation=45)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. Confusion matrix
        cm = np.array([
            [metrics.get('true_negatives', 0), metrics.get('false_positives', 0)],
            [metrics.get('false_negatives', 0), metrics.get('true_positives', 0)]
        ])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Custom metrics
        custom_metrics = ['ai_detection_rate', 'false_alarm_rate', 'expected_calibration_error']
        custom_values = [metrics.get(m, 0) for m in custom_metrics]
        
        axes[1, 1].bar(custom_metrics, custom_values, color=['purple', 'brown', 'pink'])
        axes[1, 1].set_title('Custom Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      save_path: str = None):
        """Plot ROC curve."""
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                   save_path: str = None):
        """Plot Precision-Recall curve."""
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                              n_bins: int = 10, save_path: str = None):
        """Plot calibration curve."""
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                bin_accuracies.append(y_true[in_bin].mean())
                bin_confidences.append(y_proba[in_bin].mean())
                bin_counts.append(prop_in_bin)
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        plt.figure(figsize=(8, 6))
        plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model')
        plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_metrics(self, metrics: Dict[str, float], save_path: str):
        """Save metrics to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {save_path}")
    
    def load_metrics(self, load_path: str) -> Dict[str, float]:
        """Load metrics from file."""
        with open(load_path, 'r') as f:
            metrics = json.load(f)
        
        logger.info(f"Metrics loaded from {load_path}")
        return metrics
