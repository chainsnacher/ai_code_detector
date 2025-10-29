"""
Advanced cross-validation framework for robust model evaluation.
Includes stratified k-fold, time series validation, and adversarial testing.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, TimeSeriesSplit, GroupKFold,
    cross_val_score, cross_validate
)
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from pathlib import Path
import json
import time
from collections import defaultdict

logger = logging.getLogger("ai_code_detector")

class AdvancedCrossValidator:
    """Advanced cross-validation with multiple strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize cross-validator."""
        self.config = config or {
            'n_folds': 5,
            'stratified': True,
            'shuffle': True,
            'random_state': 42,
            'test_size': 0.2,
            'validation_strategies': ['stratified_kfold', 'group_kfold', 'time_series']
        }
        
        self.validation_results = {}
        self.model_performance = defaultdict(list)
    
    def validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                      groups: np.ndarray = None, 
                      validation_strategies: List[str] = None) -> Dict[str, Any]:
        """Validate model using multiple strategies."""
        if validation_strategies is None:
            validation_strategies = self.config['validation_strategies']
        
        results = {}
        
        for strategy in validation_strategies:
            logger.info(f"Running {strategy} validation")
            
            try:
                if strategy == 'stratified_kfold':
                    strategy_results = self._stratified_kfold_validation(model, X, y)
                elif strategy == 'group_kfold':
                    if groups is not None:
                        strategy_results = self._group_kfold_validation(model, X, y, groups)
                    else:
                        logger.warning("Groups not provided for GroupKFold, skipping")
                        continue
                elif strategy == 'time_series':
                    strategy_results = self._time_series_validation(model, X, y)
                elif strategy == 'holdout':
                    strategy_results = self._holdout_validation(model, X, y)
                else:
                    logger.warning(f"Unknown validation strategy: {strategy}")
                    continue
                
                results[strategy] = strategy_results
                
            except Exception as e:
                logger.error(f"Error in {strategy} validation: {e}")
                results[strategy] = {'error': str(e)}
        
        # Aggregate results
        aggregated_results = self._aggregate_results(results)
        results['aggregated'] = aggregated_results
        
        return results
    
    def _stratified_kfold_validation(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform stratified k-fold cross-validation."""
        cv = StratifiedKFold(
            n_splits=self.config['n_folds'],
            shuffle=self.config['shuffle'],
            random_state=self.config['random_state']
        )
        
        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
            'precision_weighted': make_scorer(precision_score, average='weighted'),
            'recall_weighted': make_scorer(recall_score, average='weighted')
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, return_train_score=True
        )
        
        # Calculate additional metrics
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate detailed metrics
            fold_metrics = self._calculate_detailed_metrics(y_val, y_pred)
            fold_metrics['fold'] = fold
            fold_results.append(fold_metrics)
        
        return {
            'cv_results': cv_results,
            'fold_results': fold_results,
            'mean_scores': {metric: np.mean(scores) for metric, scores in cv_results.items()},
            'std_scores': {metric: np.std(scores) for metric, scores in cv_results.items()}
        }
    
    def _group_kfold_validation(self, model, X: np.ndarray, y: np.ndarray, 
                               groups: np.ndarray) -> Dict[str, Any]:
        """Perform group k-fold cross-validation."""
        cv = GroupKFold(n_splits=self.config['n_folds'])
        
        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
            'precision_weighted': make_scorer(precision_score, average='weighted'),
            'recall_weighted': make_scorer(recall_score, average='weighted')
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, groups=groups, cv=cv, scoring=scoring, return_train_score=True
        )
        
        # Calculate additional metrics
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate detailed metrics
            fold_metrics = self._calculate_detailed_metrics(y_val, y_pred)
            fold_metrics['fold'] = fold
            fold_metrics['unique_groups'] = len(np.unique(groups[val_idx]))
            fold_results.append(fold_metrics)
        
        return {
            'cv_results': cv_results,
            'fold_results': fold_results,
            'mean_scores': {metric: np.mean(scores) for metric, scores in cv_results.items()},
            'std_scores': {metric: np.std(scores) for metric, scores in cv_results.items()}
        }
    
    def _time_series_validation(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        cv = TimeSeriesSplit(n_splits=self.config['n_folds'])
        
        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
            'precision_weighted': make_scorer(precision_score, average='weighted'),
            'recall_weighted': make_scorer(recall_score, average='weighted')
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv, scoring=scoring, return_train_score=True
        )
        
        # Calculate additional metrics
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Calculate detailed metrics
            fold_metrics = self._calculate_detailed_metrics(y_val, y_pred)
            fold_metrics['fold'] = fold
            fold_metrics['train_size'] = len(X_train)
            fold_metrics['val_size'] = len(X_val)
            fold_results.append(fold_metrics)
        
        return {
            'cv_results': cv_results,
            'fold_results': fold_results,
            'mean_scores': {metric: np.mean(scores) for metric, scores in cv_results.items()},
            'std_scores': {metric: np.std(scores) for metric, scores in cv_results.items()}
        }
    
    def _holdout_validation(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform holdout validation."""
        from sklearn.model_selection import train_test_split
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], 
            stratify=y
        )
        
        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = self._calculate_detailed_metrics(y_val, y_pred)
        
        return {
            'holdout_results': metrics,
            'train_size': len(X_train),
            'val_size': len(X_val)
        }
    
    def _calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate detailed metrics for a single fold."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        metrics.update({
            'precision_human': precision_per_class[0] if len(precision_per_class) > 0 else 0,
            'precision_ai': precision_per_class[1] if len(precision_per_class) > 1 else 0,
            'recall_human': recall_per_class[0] if len(recall_per_class) > 0 else 0,
            'recall_ai': recall_per_class[1] if len(recall_per_class) > 1 else 0,
            'f1_human': f1_per_class[0] if len(f1_per_class) > 0 else 0,
            'f1_ai': f1_per_class[1] if len(f1_per_class) > 1 else 0,
        })
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            })
        
        return metrics
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results across validation strategies."""
        aggregated = {
            'strategy_scores': {},
            'overall_performance': {},
            'consistency_metrics': {}
        }
        
        # Extract scores for each strategy
        for strategy, strategy_results in results.items():
            if 'error' not in strategy_results:
                if 'mean_scores' in strategy_results:
                    aggregated['strategy_scores'][strategy] = strategy_results['mean_scores']
                elif 'holdout_results' in strategy_results:
                    aggregated['strategy_scores'][strategy] = strategy_results['holdout_results']
        
        # Calculate overall performance
        if aggregated['strategy_scores']:
            all_scores = list(aggregated['strategy_scores'].values())
            
            # Average across strategies
            for metric in all_scores[0].keys():
                values = [scores.get(metric, 0) for scores in all_scores]
                aggregated['overall_performance'][f'{metric}_mean'] = np.mean(values)
                aggregated['overall_performance'][f'{metric}_std'] = np.std(values)
                aggregated['overall_performance'][f'{metric}_min'] = np.min(values)
                aggregated['overall_performance'][f'{metric}_max'] = np.max(values)
        
        # Calculate consistency metrics
        if len(aggregated['strategy_scores']) > 1:
            for metric in aggregated['overall_performance'].keys():
                if metric.endswith('_mean'):
                    base_metric = metric.replace('_mean', '')
                    values = [scores.get(base_metric, 0) for scores in aggregated['strategy_scores'].values()]
                    if len(values) > 1:
                        aggregated['consistency_metrics'][f'{base_metric}_cv'] = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        return aggregated
    
    def compare_models(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray, 
                      groups: np.ndarray = None) -> Dict[str, Any]:
        """Compare multiple models using cross-validation."""
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Validating model: {model_name}")
            
            try:
                results = self.validate_model(model, X, y, groups)
                comparison_results[model_name] = results
                
            except Exception as e:
                logger.error(f"Error validating {model_name}: {e}")
                comparison_results[model_name] = {'error': str(e)}
        
        # Create comparison summary
        comparison_summary = self._create_comparison_summary(comparison_results)
        
        return {
            'model_results': comparison_results,
            'comparison_summary': comparison_summary
        }
    
    def _create_comparison_summary(self, comparison_results: Dict[str, Any]) -> pd.DataFrame:
        """Create comparison summary DataFrame."""
        summary_data = []
        
        for model_name, results in comparison_results.items():
            if 'error' not in results:
                # Extract performance metrics
                if 'aggregated' in results and 'overall_performance' in results['aggregated']:
                    perf = results['aggregated']['overall_performance']
                    summary_data.append({
                        'Model': model_name,
                        'Accuracy': perf.get('accuracy_mean', 0),
                        'F1_Score': perf.get('f1_score_mean', 0),
                        'Precision': perf.get('precision_mean', 0),
                        'Recall': perf.get('recall_mean', 0),
                        'Accuracy_Std': perf.get('accuracy_std', 0),
                        'F1_Std': perf.get('f1_score_std', 0),
                    })
                else:
                    summary_data.append({
                        'Model': model_name,
                        'Accuracy': 0,
                        'F1_Score': 0,
                        'Precision': 0,
                        'Recall': 0,
                        'Accuracy_Std': 0,
                        'F1_Std': 0,
                    })
            else:
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': 0,
                    'F1_Score': 0,
                    'Precision': 0,
                    'Recall': 0,
                    'Accuracy_Std': 0,
                    'F1_Std': 0,
                })
        
        return pd.DataFrame(summary_data)
    
    def plot_validation_results(self, results: Dict[str, Any], save_path: str = None):
        """Plot validation results."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract data for plotting
        strategies = []
        metrics = []
        scores = []
        errors = []
        
        for strategy, strategy_results in results.items():
            if 'error' not in strategy_results and 'mean_scores' in strategy_results:
                for metric, score in strategy_results['mean_scores'].items():
                    if metric.endswith('_score') or metric == 'accuracy':
                        strategies.append(strategy)
                        metrics.append(metric)
                        scores.append(score)
                        
                        # Add error bars if available
                        if 'std_scores' in strategy_results:
                            errors.append(strategy_results['std_scores'].get(metric, 0))
                        else:
                            errors.append(0)
        
        if not strategies:
            logger.warning("No valid results to plot")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by metric
        unique_metrics = list(set(metrics))
        x_pos = np.arange(len(unique_metrics))
        width = 0.8 / len(set(strategies))
        
        for i, strategy in enumerate(set(strategies)):
            strategy_scores = [scores[j] for j, s in enumerate(strategies) if s == strategy]
            strategy_errors = [errors[j] for j, s in enumerate(strategies) if s == strategy]
            
            ax.bar(x_pos + i * width, strategy_scores, width, 
                  label=strategy, yerr=strategy_errors, capsize=5)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Cross-Validation Results by Strategy')
        ax.set_xticks(x_pos + width * (len(set(strategies)) - 1) / 2)
        ax.set_xticklabels(unique_metrics, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """Save validation results to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Validation results saved to {save_path}")
    
    def load_results(self, load_path: str) -> Dict[str, Any]:
        """Load validation results from file."""
        with open(load_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Validation results loaded from {load_path}")
        return results
