# Diagnostic runner for detection
import sys, os, json
sys.path.append('src')
from preprocessing.feature_extractor import StatisticalFeatureExtractor
from models.ensemble_model import AdvancedEnsembleDetector
from models.baseline_models import BaselineModelTrainer
from utils.data_utils import CodePreprocessor
import joblib
import pandas as pd
import numpy as np

sample_path = 'data/train/ai/py_ai_ext_0000.py'
print('Sample file:', sample_path, 'exists?', os.path.exists(sample_path))
with open(sample_path, 'r', encoding='utf-8', errors='ignore') as f:
    code = f.read()
print('Sample length (chars):', len(code))

# Clean and extract
clean = CodePreprocessor.clean_code(code)
extractor = StatisticalFeatureExtractor()
features_dict = extractor.extract_features(clean, language='python')
features_df = pd.DataFrame([features_dict]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
print('Extracted features shape:', features_df.shape)

# canonical cols
canonical_cols = None
if os.path.exists('models/feature_columns.json'):
    with open('models/feature_columns.json') as f:
        canonical_cols = json.load(f)
    print('Canonical cols loaded, len =', len(canonical_cols))
else:
    print('No models/feature_columns.json found')

# Align
if canonical_cols is not None:
    try:
        features_df = features_df.reindex(columns=canonical_cols, fill_value=0)
    except Exception as e:
        print('Reindex fail:', e)

X = features_df.values.astype(float)
print('X.shape after align:', X.shape)

# Load baseline models
baseline = BaselineModelTrainer()
try:
    baseline.load_models('models/baseline')
    print('Loaded baseline models:', list(baseline.models.keys()))
except Exception as e:
    print('Could not load baseline models:', e)

# Load ensemble
detector = AdvancedEnsembleDetector()
try:
    detector.load_ensemble('models/ensemble')
    print('Loaded ensemble meta classifier:', detector.meta_classifier is not None)
except Exception as e:
    print('Could not load ensemble:', e)

# Attach baseline models from baseline.trainer if loaded
if hasattr(baseline, 'models') and baseline.models:
    for name, model in baseline.models.items():
        try:
            detector.add_base_model(name, model)
        except Exception:
            detector.base_models[name] = model

# Per-model preds
preds = {}
probs = {}
for name, model in detector.base_models.items():
    print('\nModel:', name)
    print(' n_features_in_ =', getattr(model, 'n_features_in_', None))
    try:
        X_model = X
        expected = getattr(model, 'n_features_in_', None)
        if expected is not None and X.shape[1] != expected:
            print('  -> feature count mismatch: expected', expected, 'provided', X.shape[1])
            if X.shape[1] > expected:
                X_model = X[:, :expected]
                print('  -> trimmed to', X_model.shape)
            else:
                X_model = np.hstack([X, np.zeros((X.shape[0], expected - X.shape[1]))])
                print('  -> padded to', X_model.shape)
        p = model.predict(X_model)
        preds[name] = p
        print('  predict:', p)
        if hasattr(model, 'predict_proba'):
            pr = model.predict_proba(X_model)
            probs[name] = pr
            print('  predict_proba shape:', pr.shape)
    except Exception as e:
        print('  prediction failed:', e)

# Meta features and ensemble decision
try:
    meta_feats = detector.meta_feature_generator.generate_meta_features(preds, probs)
    print('\nMeta features shape:', meta_feats.shape)
    if detector.scaler is not None and detector.meta_classifier is not None:
        meta_scaled = detector.scaler.transform(meta_feats)
        pred_arr = detector.meta_classifier.predict(meta_scaled)
        print('Meta classifier predict:', pred_arr)
        if hasattr(detector.meta_classifier, 'predict_proba'):
            mp = detector.meta_classifier.predict_proba(meta_scaled)
            print('Meta classifier proba:', mp)
    else:
        try:
            print('Detector.predict fallback:', detector.predict(X))
        except Exception as e:
            print('Detector.predict failed:', e)
except Exception as e:
    print('Meta feature / ensemble step failed:', e)

print('\nDone')
