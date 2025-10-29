"""
Compute SHAP explanations for available tree-based baseline models and export CSVs for Power BI.
- Loads canonical feature columns from models/feature_columns.json
- Loads data/processed/features.csv
- Loads baseline models from models/baseline
- For tree-based models (RandomForest, GradientBoosting) runs SHAP TreeExplainer (if shap available) on a sample subset (default 200 rows)
- Exports per-sample SHAP values CSV and mean absolute shap per feature CSV to data/powerbi/shap/

Usage:
  python -c "import sys; sys.path.insert(0,'.'); import runpy; runpy.run_path('scripts/compute_shap_and_export.py', run_name='__main__')"

If shap is not installed, the script falls back to a simple feature importance export (model.feature_importances_).
"""

import sys
import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ai_code_detector')

# Ensure imports from src work
sys.path.insert(0, '.')

# Alias old module paths to current ones so joblib/pickle can unpickle legacy artifacts
import importlib
try:
    sys.modules['models.baseline_models'] = importlib.import_module('src.models.baseline_models')
except Exception:
    pass
try:
    sys.modules['models.ensemble_model'] = importlib.import_module('src.models.ensemble_model')
except Exception:
    pass

from src.models.baseline_models import BaselineModelTrainer

FEATURES_CSV = Path('data/processed/features.csv')
FEATURE_COLUMNS_FILE = Path('models/feature_columns.json')
BASELINE_DIR = Path('models/baseline')
SHAP_OUT_DIR = Path('data/powerbi/shap')
SHAP_OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 200  # limit rows for SHAP to keep runtime modest


def try_import_shap():
    try:
        import shap
        return shap
    except Exception:
        return None


def load_feature_columns():
    if FEATURE_COLUMNS_FILE.exists():
        return json.load(open(FEATURE_COLUMNS_FILE, 'r', encoding='utf-8'))
    else:
        logger.warning('Feature columns JSON not found; using all CSV columns')
        df_tmp = pd.read_csv(FEATURES_CSV, nrows=1)
        return list(df_tmp.columns)


def main():
    logger.info('Starting SHAP compute and export')

    if not FEATURES_CSV.exists():
        logger.error(f'Features CSV missing: {FEATURES_CSV}')
        return

    df = pd.read_csv(FEATURES_CSV)
    feature_columns = load_feature_columns()

    # Guard: ensure feature_columns are in df
    feature_columns = [c for c in feature_columns if c in df.columns]

    X = df[feature_columns]
    n = len(X)
    logger.info(f'Loaded features: n_samples={n}, n_features={len(feature_columns)}')

    trainer = BaselineModelTrainer()
    try:
        trainer.load_models(str(BASELINE_DIR))
    except Exception as e:
        logger.warning(f'Could not load baseline models using trainer.load_models(): {e}')
        # attempt to load joblib directly (best-effort)
        import joblib
        for p in BASELINE_DIR.glob('*_model.pkl'):
            name = p.stem.replace('_model', '')
            try:
                trainer.models[name] = joblib.load(p)
                logger.info(f'Loaded model {name} from {p}')
            except Exception as ee:
                logger.warning(f'Failed to load {p}: {ee}')

    shap_lib = try_import_shap()
    if shap_lib is None:
        logger.warning('shap library not found; will export feature_importances_ where available')

    # Process tree-based models
    for model_name, model in trainer.models.items():
        logger.info(f'Processing model: {model_name}')

        # prefer tree models
        is_tree = False
        try:
            clf = model.named_steps['classifier'] if hasattr(model, 'named_steps') else model
        except Exception:
            clf = model

        # detect tree models with feature_importances_
        if hasattr(clf, 'feature_importances_'):
            is_tree = True

        if not is_tree:
            logger.info(f'Skipping {model_name}: not a tree-based model (no feature_importances_)')
            continue

        # select sample
        sample_idx = np.arange(n)
        if n > SAMPLE_SIZE:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(n, size=SAMPLE_SIZE, replace=False)

        X_sample = X.iloc[sample_idx]

        if shap_lib is not None:
            try:
                # If model is a pipeline, get classifier
                if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                    model_for_shap = model.named_steps['classifier']
                    # For scikit-learn pipeline with scaler, we need to pass scaled input to explainer
                    # We'll create a small wrapper that applies pipeline up to classifier
                    def predict_fn(x):
                        # x is numpy array
                        return model.predict_proba(x)

                    # If pipeline sanitizes/scales, apply pipeline.transform on raw X
                    # Here we'll just use the pipeline's predict_proba with raw df input to avoid mistakes
                    explainer = shap_lib.TreeExplainer(model_for_shap)
                    shap_values = explainer.shap_values(model_for_shap.transform(X_sample) if hasattr(model_for_shap, 'transform') else X_sample)
                else:
                    explainer = shap_lib.TreeExplainer(clf)
                    shap_values = explainer.shap_values(X_sample)

                # shap_values shape depends on binary vs multiclass
                # For binary classification shap_values is list of two arrays or single 2D array; normalize to 2D for positive class
                if isinstance(shap_values, list) and len(shap_values) >= 2:
                    # choose positive class index 1
                    shap_pos = shap_values[1]
                else:
                    shap_pos = shap_values

                # Build per-sample dataframe: columns = feature_columns
                shap_df = pd.DataFrame(shap_pos, columns=feature_columns)
                out_file = SHAP_OUT_DIR / f'{model_name}_shap_samples.csv'
                shap_df.to_csv(out_file, index=False)
                logger.info(f'Wrote per-sample SHAP for {model_name} to {out_file}')

                # Mean absolute shap per feature
                mean_abs = np.abs(shap_df).mean(axis=0)
                mean_df = pd.DataFrame({ 'feature': mean_abs.index, 'mean_abs_shap': mean_abs.values })
                mean_out = SHAP_OUT_DIR / f'{model_name}_shap_summary.csv'
                mean_df.to_csv(mean_out, index=False)
                logger.info(f'Wrote SHAP summary for {model_name} to {mean_out}')

            except Exception as e:
                logger.warning(f'Error computing SHAP for {model_name}: {e}')
                # fallback to feature_importances_
                try:
                    fi = clf.feature_importances_
                    fi_df = pd.DataFrame({ 'feature': feature_columns, 'importance': fi })
                    out_file = SHAP_OUT_DIR / f'{model_name}_feature_importances.csv'
                    fi_df.to_csv(out_file, index=False)
                    logger.info(f'Wrote fallback feature_importances_ for {model_name} to {out_file}')
                except Exception as ee:
                    logger.warning(f'Failed fallback feature_importances_ for {model_name}: {ee}')
        else:
            # No shap - export feature_importances_
            try:
                fi = clf.feature_importances_
                fi_df = pd.DataFrame({ 'feature': feature_columns, 'importance': fi })
                out_file = SHAP_OUT_DIR / f'{model_name}_feature_importances.csv'
                fi_df.to_csv(out_file, index=False)
                logger.info(f'Wrote fallback feature_importances_ for {model_name} to {out_file}')
            except Exception as ee:
                logger.warning(f'No shap and failed to fetch feature_importances_ for {model_name}: {ee}')

    logger.info('SHAP compute and export complete')


if __name__ == '__main__':
    main()
