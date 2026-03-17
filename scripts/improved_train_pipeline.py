#!/usr/bin/env python3
"""
Improved training pipeline with:
- Better class balancing
- Enhanced feature engineering
- Proper train/validation/test splits
- Class imbalance handling
- Comprehensive evaluation
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, accuracy_score
)
import joblib
import json
import logging
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.feature_extractor import StatisticalFeatureExtractor
from src.preprocessing.code_tokenizer import AdvancedCodeTokenizer
from src.preprocessing.ast_parser import ASTFeatureExtractor
from src.models.baseline_models import BaselineModelTrainer
from src.utils.data_utils import DataProcessor
from src.models.ensemble_model import AdvancedEnsembleDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_training_data(data_dir: Path = None) -> pd.DataFrame:
    """Load training data from code files."""
    if data_dir is None:
        data_dir = ROOT / 'data' / 'train'
    
    human_dir = data_dir / 'human'
    ai_dir = data_dir / 'ai'
    
    if not human_dir.exists() or not ai_dir.exists():
        raise FileNotFoundError(
            f"Training directories not found. Expected:\n"
            f"  {human_dir}\n"
            f"  {ai_dir}\n"
            f"Run scripts/collect_real_data.py first to generate training data."
        )
    
    logger.info("Loading training data...")
    
    # Load human code
    human_files = list(human_dir.glob('*.py')) + list(human_dir.glob('*.js')) + list(human_dir.glob('*.java'))
    ai_files = list(ai_dir.glob('*.py')) + list(ai_dir.glob('*.js')) + list(ai_dir.glob('*.java'))
    
    logger.info(f"Found {len(human_files)} human files and {len(ai_files)} AI files")
    
    if len(human_files) == 0 or len(ai_files) == 0:
        raise ValueError(
            "No training files found. Please run scripts/collect_real_data.py first "
            "or ensure data/train/human/ and data/train/ai/ contain code files."
        )
    
    # Extract features
    feature_extractor = StatisticalFeatureExtractor()
    ast_extractor = ASTFeatureExtractor()
    tokenizer = AdvancedCodeTokenizer()
    
    features_list = []
    labels = []
    code_samples = []
    
    # Process human files (label = 0)
    logger.info("Processing human code files...")
    for i, filepath in enumerate(human_files):
        try:
            code = filepath.read_text(encoding='utf-8', errors='ignore')
            if len(code.strip()) < 50:  # Skip very short files
                continue
            
            # Detect language
            lang = 'python'
            if filepath.suffix == '.js':
                lang = 'javascript'
            elif filepath.suffix == '.java':
                lang = 'java'
            
            # Extract features
            features = {}
            try:
                features.update(feature_extractor.extract_features(code, language=lang))
            except:
                pass
            try:
                ast_features = ast_extractor.extract_features(code, lang)
                if ast_features:
                    features.update(ast_features)
            except:
                pass
            try:
                token_features = tokenizer.get_code_metrics(code, lang)
                if token_features:
                    features.update(token_features)
            except:
                pass
            
            if features:
                features['code_file'] = str(filepath)
                features_list.append(features)
                labels.append(0)  # Human
                code_samples.append(code[:200])  # Store sample
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(human_files)} human files")
        
        except Exception as e:
            logger.debug(f"Error processing {filepath}: {e}")
            continue
    
    # Process AI files (label = 1)
    logger.info("Processing AI code files...")
    for i, filepath in enumerate(ai_files):
        try:
            code = filepath.read_text(encoding='utf-8', errors='ignore')
            if len(code.strip()) < 50:
                continue
            
            lang = 'python'
            if filepath.suffix == '.js':
                lang = 'javascript'
            elif filepath.suffix == '.java':
                lang = 'java'
            
            features = {}
            try:
                features.update(feature_extractor.extract_features(code, language=lang))
            except:
                pass
            try:
                ast_features = ast_extractor.extract_features(code, lang)
                if ast_features:
                    features.update(ast_features)
            except:
                pass
            try:
                token_features = tokenizer.get_code_metrics(code, lang)
                if token_features:
                    features.update(token_features)
            except:
                pass
            
            if features:
                features['code_file'] = str(filepath)
                features_list.append(features)
                labels.append(1)  # AI
                code_samples.append(code[:200])
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(ai_files)} AI files")
        
        except Exception as e:
            logger.debug(f"Error processing {filepath}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    df['label'] = labels
    df['code_sample'] = code_samples
    
    logger.info(f"Loaded {len(df)} samples with {len(df.columns) - 2} features")
    logger.info(f"Class distribution: Human={sum(labels==0)}, AI={sum(labels==1)}")
    
    return df


def balance_classes(df: pd.DataFrame, method: str = 'undersample') -> pd.DataFrame:
    """Balance classes using undersampling or oversampling."""
    labels = df['label'].values
    human_count = sum(labels == 0)
    ai_count = sum(labels == 1)
    
    logger.info(f"Before balancing: Human={human_count}, AI={ai_count}")
    
    if method == 'undersample':
        # Undersample majority class
        min_count = min(human_count, ai_count)
        human_indices = df[df['label'] == 0].index[:min_count]
        ai_indices = df[df['label'] == 1].index[:min_count]
        balanced_df = pd.concat([df.loc[human_indices], df.loc[ai_indices]])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    elif method == 'oversample':
        # Oversample minority class
        from sklearn.utils import resample
        human_df = df[df['label'] == 0]
        ai_df = df[df['label'] == 1]
        
        if human_count < ai_count:
            human_df = resample(human_df, replace=True, n_samples=ai_count, random_state=42)
        else:
            ai_df = resample(ai_df, replace=True, n_samples=human_count, random_state=42)
        
        balanced_df = pd.concat([human_df, ai_df])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    else:
        balanced_df = df
    
    logger.info(f"After balancing: Human={sum(balanced_df['label']==0)}, AI={sum(balanced_df['label']==1)}")
    
    return balanced_df


def train_improved_models(df: pd.DataFrame, output_dir: Path = None):
    """Train improved models with proper evaluation."""
    if output_dir is None:
        output_dir = ROOT / 'models'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Balance classes
    df_balanced = balance_classes(df, method='undersample')
    
    # Prepare features
    non_feature_cols = {'code_file', 'label', 'code_sample'}
    feature_cols = [c for c in df_balanced.columns if c not in non_feature_cols]
    
    X = df_balanced[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = df_balanced['label'].values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Class distribution: {Counter(y)}")
    
    # Train/validation/test split (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Save feature columns for inference
    feature_columns_file = output_dir / 'feature_columns.json'
    with open(feature_columns_file, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    logger.info(f"Saved feature columns to {feature_columns_file}")
    
    # Train baseline models with class weights
    logger.info("Training baseline models...")
    trainer = BaselineModelTrainer()
    
    # Update model configs to handle class imbalance
    trainer.model_configs['random_forest']['class_weight'] = 'balanced_subsample'
    trainer.model_configs['svm']['class_weight'] = 'balanced'
    trainer.model_configs['logistic_regression']['class_weight'] = 'balanced'
    
    results = trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on validation set
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SET PERFORMANCE")
    logger.info("="*60)
    
    val_results = {}
    for model_name, model in trainer.models.items():
        try:
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba) if y_proba is not None else None
            
            val_results[model_name] = {
                'accuracy': acc,
                'f1_score': f1,
                'roc_auc': auc
            }
            
            logger.info(f"{model_name}:")
            logger.info(f"  Accuracy: {acc:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  ROC-AUC: {auc:.4f}" if auc else "  ROC-AUC: N/A")
            logger.info(classification_report(y_val, y_pred, target_names=['Human', 'AI']))
        
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
    
    # Test set evaluation
    logger.info("\n" + "="*60)
    logger.info("TEST SET PERFORMANCE")
    logger.info("="*60)
    
    test_results = {}
    for model_name, model in trainer.models.items():
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            
            test_results[model_name] = {
                'accuracy': acc,
                'f1_score': f1,
                'roc_auc': auc
            }
            
            logger.info(f"{model_name}:")
            logger.info(f"  Accuracy: {acc:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            logger.info(f"  ROC-AUC: {auc:.4f}" if auc else "  ROC-AUC: N/A")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"  Confusion Matrix:\n{cm}")
        
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
    
    # Create and train ensemble
    logger.info("\n" + "="*60)
    logger.info("TRAINING ENSEMBLE")
    logger.info("="*60)
    
    try:
        ensemble = trainer.create_ensemble(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
        
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        ensemble_f1 = f1_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)
        
        logger.info(f"Ensemble Performance:")
        logger.info(f"  Accuracy: {ensemble_acc:.4f}")
        logger.info(f"  F1-Score: {ensemble_f1:.4f}")
        logger.info(f"  ROC-AUC: {ensemble_auc:.4f}")
        
        test_results['ensemble'] = {
            'accuracy': ensemble_acc,
            'f1_score': ensemble_f1,
            'roc_auc': ensemble_auc
        }
    
    except Exception as e:
        logger.error(f"Error creating ensemble: {e}")
    
    # Train advanced ensemble
    try:
        adv_ensemble = AdvancedEnsembleDetector()
        for name, model in trainer.models.items():
            if name != 'ensemble':
                adv_ensemble.add_base_model(name, model)
        
        adv_ensemble.fit(X_train, y_train, X_val, y_val)
        adv_ensemble.save_ensemble(str(output_dir / 'ensemble'))
        
        adv_pred = adv_ensemble.predict(X_test)
        adv_acc = accuracy_score(y_test, adv_pred)
        adv_f1 = f1_score(y_test, adv_pred)
        
        logger.info(f"Advanced Ensemble Performance:")
        logger.info(f"  Accuracy: {adv_acc:.4f}")
        logger.info(f"  F1-Score: {adv_f1:.4f}")
        
        test_results['advanced_ensemble'] = {
            'accuracy': adv_acc,
            'f1_score': adv_f1
        }
    
    except Exception as e:
        logger.error(f"Error training advanced ensemble: {e}")
    
    # Save models
    baseline_dir = output_dir / 'baseline'
    trainer.save_models(str(baseline_dir))
    
    # Save results
    results_file = output_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'validation_results': val_results,
            'test_results': test_results,
            'feature_count': len(feature_cols),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }, f, indent=2)
    
    logger.info(f"\nModels saved to {output_dir}")
    logger.info(f"Results saved to {results_file}")
    
    return trainer, test_results


def main():
    """Main training function."""
    logger.info("="*60)
    logger.info("IMPROVED TRAINING PIPELINE")
    logger.info("="*60)
    
    # Load data
    try:
        df = load_training_data()
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        logger.info("\nTo generate training data, run:")
        logger.info("  python scripts/collect_real_data.py")
        return
    
    # Save features CSV for compatibility
    processed_dir = ROOT / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_file = processed_dir / 'features.csv'
    
    # Remove code_sample for CSV (keep only first 200 chars if needed)
    df_csv = df.copy()
    if 'code_sample' in df_csv.columns:
        df_csv = df_csv.drop('code_sample', axis=1)
    
    df_csv.to_csv(features_file, index=False)
    logger.info(f"Saved features CSV to {features_file}")
    
    # Train models
    trainer, results = train_improved_models(df)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info("Best performing models on test set:")
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1].get('f1_score', 0), reverse=True):
        logger.info(f"  {model_name}: F1={metrics.get('f1_score', 0):.4f}, Acc={metrics.get('accuracy', 0):.4f}")


if __name__ == '__main__':
    main()
