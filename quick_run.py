"""
Quick run script for AI Code Detection System - skips heavy computations
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging

# Setup path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logger
from src.utils.config import get_config
from src.models.baseline_models import BaselineModelTrainer
from src.models.ensemble_model import AdvancedEnsembleDetector

logger = setup_logger()
config = get_config()

def main():
    """Quick run with pre-computed features."""
    logger.info("Starting Quick AI Code Detection Pipeline")
    
    # Check if features exist
    features_path = Path('data/processed/features.csv')
    if not features_path.exists():
        logger.error(f"Features file not found at {features_path}")
        logger.info("Run: python scripts/ingest_samples.py --split train --out data/processed/features.csv")
        return
    
    # Load features
    logger.info("Loading pre-computed features...")
    df = pd.read_csv(features_path)
    
    # Separate features and labels
    label_col = 'label' if 'label' in df.columns else None
    if label_col is None:
        logger.error("No 'label' column found in features")
        return
    
    y = df[label_col].values
    X = df.drop(columns=[label_col, 'code'] if 'code' in df.columns else [label_col]).values
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Train baseline models
    logger.info("Step 1: Training Baseline Models")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    baseline_trainer = BaselineModelTrainer(config.get_section('models'))
    baseline_results = baseline_trainer.train_models(X_train, y_train, X_test, y_test)
    logger.info(f"Baseline training completed. Results: {baseline_results}")
    
    # Create ensemble
    logger.info("Step 2: Creating Ensemble Model")
    ensemble_detector = AdvancedEnsembleDetector(config.get_section('models'))
    
    for model_name, model_obj in baseline_trainer.models.items():
        ensemble_detector.add_base_model(f"baseline_{model_name}", model_obj)
    
    ensemble_detector.fit(X_train, y_train)
    
    # Evaluate ensemble
    logger.info("Step 3: Evaluating Ensemble")
    y_pred = ensemble_detector.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Ensemble Results:")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall: {rec:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    
    # Save ensemble
    logger.info("Saving ensemble model...")
    ensemble_detector.save_ensemble('models/ensemble')
    
    logger.info("✅ Quick pipeline completed successfully!")
    print("✅ Pipeline complete! Models saved to models/ensemble/")

if __name__ == '__main__':
    main()
