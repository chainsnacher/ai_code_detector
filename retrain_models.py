#!/usr/bin/env python3
"""
Improved training with better class separation
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path
from src.preprocessing.feature_extractor import StatisticalFeatureExtractor
from src.preprocessing.code_tokenizer import AdvancedCodeTokenizer
from src.preprocessing.ast_parser import ASTFeatureExtractor
from src.models.baseline_models import BaselineModelTrainer
from src.utils.data_utils import DataProcessor
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('improve_model')

# Load current features
df = pd.read_csv('data/processed/features.csv')
print(f"Current features: {df.shape}")

# Load canonical columns
with open('models/feature_columns.json', 'r') as f:
    canonical_cols = json.load(f)

print(f"Canonical columns: {len(canonical_cols)}")

# Prepare data with proper train-test split
from sklearn.model_selection import train_test_split

# Get features and labels
label_col = 'label'

# Remove non-numeric columns from dataframe
non_feature_cols = {'code_file', 'label'}
available_features = [c for c in df.columns if c not in non_feature_cols and c in canonical_cols]
print(f"Using {len(available_features)} features")

X = df[available_features].values.astype(float)
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
y = df[label_col].values

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train distribution: AI={sum(y_train==1)}, Human={sum(y_train==0)}")
print(f"Test distribution: AI={sum(y_test==1)}, Human={sum(y_test==0)}")

# Train models
logger.info("Training models...")
trainer = BaselineModelTrainer()
results = trainer.train_models(X_train, y_train)

# Create voting ensemble
logger.info("Creating voting ensemble...")
try:
    ensemble = trainer.create_ensemble(X_train, y_train)
    logger.info("Voting ensemble created successfully")
except Exception as e:
    logger.warning(f"Could not create voting ensemble: {e}")

# Evaluate on test set
print(f"\n{'='*60}")
print("MODEL PERFORMANCE ON TEST SET")
print(f"{'='*60}")

for name, model in trainer.models.items():
    try:
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).sum() / len(y_test) * 100
        print(f"{name}: {accuracy:.1f}% accuracy")
    except Exception as e:
        print(f"{name}: Error - {e}")

# Save models
logger.info("Saving models...")
trainer.save_models('models/baseline')

print(f"\n✓ Models retrained and saved!")
