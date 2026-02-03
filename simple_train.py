"""
Simplified training script for AI Code Detection System
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logger

logger = setup_logger()

def main():
    """Simple training on custom data."""
    logger.info("Starting Simplified Training Pipeline")
    
    # Check if features exist
    features_path = Path('data/processed/features.csv')
    if not features_path.exists():
        logger.error(f"Features file not found at {features_path}")
        return
    
    # Load features
    logger.info("Loading features...")
    df = pd.read_csv(features_path)
    
    # Get label column
    label_col = 'label'
    if label_col not in df.columns:
        logger.error(f"No '{label_col}' column found in features")
        return
    
    y = df[label_col].values
    # Drop non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    X = df[numeric_cols].values
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Classes: AI={sum(y==1)}, Human={sum(y==0)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {}
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    y_pred = rf.predict(X_test)
    logger.info(f"Random Forest - Acc: {accuracy_score(y_test, y_pred):.4f}, " + 
                f"F1: {f1_score(y_test, y_pred):.4f}")
    
    # Train Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr
    
    y_pred = lr.predict(X_test)
    logger.info(f"Logistic Regression - Acc: {accuracy_score(y_test, y_pred):.4f}, " +
                f"F1: {f1_score(y_test, y_pred):.4f}")
    
    # Create ensemble with voting
    from sklearn.ensemble import VotingClassifier
    
    logger.info("Creating Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[('rf', models['random_forest']), ('lr', models['logistic_regression'])],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Ensemble Results:")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall: {rec:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    
    # Save models
    import joblib
    models_dir = Path('models/simple_ensemble')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(ensemble, models_dir / 'voting_classifier.pkl')
    joblib.dump(rf, models_dir / 'random_forest.pkl')
    joblib.dump(lr, models_dir / 'logistic_regression.pkl')
    
    logger.info(f"Models saved to {models_dir}")
    logger.info("✅ Training completed successfully!")
    print("\n✅ TRAINING COMPLETE!")
    print(f"✅ Ensemble Accuracy: {acc:.2%}")
    print(f"✅ Models saved to: {models_dir}")

if __name__ == '__main__':
    main()
