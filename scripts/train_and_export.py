"""
Train baseline models and ensemble, persist canonical feature columns, and export Power BI CSVs.

This script is conservative:
- If a label column isn't found in the processed features, it will save the canonical feature list and exit with a clear message.
- If labels are present it will train baseline models, assemble the ensemble, save artifacts in models/, and export Power BI-ready CSVs to data/powerbi/.

Usage: python scripts/train_and_export.py
"""

import json
import logging
from pathlib import Path
import sys

# Ensure 'src' is importable when invoked as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd

from src.utils.data_utils import DataProcessor
from src.models.baseline_models import BaselineModelTrainer
from src.models.ensemble_model import AdvancedEnsembleDetector
from src.utils.powerbi_exporter import PowerBIExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_code_detector")

FEATURES_CSV = Path("data/processed/features.csv")
FEATURE_COLUMNS_FILE = Path("models/feature_columns.json")
BASELINE_SAVE_DIR = Path("models/baseline")
ENSEMBLE_SAVE_DIR = Path("models/ensemble")


def main():
    logger.info("Starting train_and_export script")

    if not FEATURES_CSV.exists():
        logger.error(f"Features file not found: {FEATURES_CSV}")
        return

    df = pd.read_csv(FEATURES_CSV)
    logger.info(f"Loaded features CSV with shape: {df.shape}")

    # Detect label column
    label_col = None
    for candidate in ["label", "Label", "target", "y", "labelled"]:
        if candidate in df.columns:
            label_col = candidate
            break

    # Determine canonical feature columns (exclude obvious non-feature fields)
    non_feature_candidates = {
        "id", "code_sample", "label", "Label", "target", "y", "hash", "code_hash",
        "filepath", "filename", "language", "language_confidence"
    }
    feature_columns = [c for c in df.columns if c not in non_feature_candidates]

    # Persist canonical feature list
    FEATURE_COLUMNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_COLUMNS_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)
    logger.info(f"Saved canonical feature columns to {FEATURE_COLUMNS_FILE} ({len(feature_columns)} columns)")

    if label_col is None:
        logger.warning(
            "No label column found in features CSV. Aborting full training.\n"
            "To train models you must provide a labelled dataset (a column named 'label' is recommended).\n"
            "You can re-run this script after adding labels or pass a labelled CSV to an extended version of this script."
        )
        return

    # Prepare data
    processor = DataProcessor()

    # If labels are numeric 0/1, keep as is; else use label encoder
    labels_raw = df[label_col].tolist()
    y = processor.prepare_labels(labels_raw) if df[label_col].dtype == object else df[label_col].values

    # Coerce features to numeric and sanitize infinities/NaNs
    import numpy as _np
    import pandas as _pd
    df[feature_columns] = df[feature_columns].apply(_pd.to_numeric, errors='coerce')
    df[feature_columns] = df[feature_columns].replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)

    X = processor.prepare_features(df, feature_columns)

    # Train baseline models
    trainer = BaselineModelTrainer()
    logger.info("Training baseline models (this may take a while)")
    results = trainer.train_models(X, y)

    # Save baseline artifacts
    trainer.save_models(str(BASELINE_SAVE_DIR))

    # Build ensemble using BaselineModelTrainer's create_ensemble (voting classifier)
    try:
        ensemble = trainer.create_ensemble(X, y)
        # Save the voting ensemble into baseline folder as well
        BASELINE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        from joblib import dump
        dump(ensemble, BASELINE_SAVE_DIR / "voting_ensemble_model.pkl")
        logger.info(f"Saved voting ensemble to {BASELINE_SAVE_DIR / 'voting_ensemble_model.pkl'}")
    except Exception as e:
        logger.warning(f"Could not create voting ensemble: {e}")

    # Fit advanced ensemble detector by loading trained base models
    adv = AdvancedEnsembleDetector()
    for model_name, model in trainer.models.items():
        adv.add_base_model(model_name, model)

    adv.fit(X, y)
    adv.save_ensemble(str(ENSEMBLE_SAVE_DIR))

    # Prepare predictions for Power BI export (use the ensemble detector to predict)
    try:
        preds = adv.predict(X)
    except Exception:
        # Fall back to trainer ensemble or one of the models
        try:
            preds = trainer.models.get('ensemble').predict(X)
        except Exception:
            preds = trainer.models[next(iter(trainer.models))].predict(X)

    # Build predictions list for PowerBIExporter
    predictions_list = []
    from datetime import datetime
    for i, pred in enumerate(preds):
        predictions_list.append({
            'id': i,
            'code_hash': '',
            'language': '',
            'prediction': int(pred),
            'confidence': 0.0,
            'model_name': 'advanced_ensemble',
            'timestamp': datetime.now().isoformat(),
            'code_sample': ''
        })

    exporter = PowerBIExporter()

    # Feature importance: convert numeric arrays to feature->importance dict per model
    feature_importance_out = {}
    for model_name, imp in trainer.feature_importance.items():
        if hasattr(imp, 'tolist'):
            imp_arr = list(imp)
        else:
            imp_arr = []
        # Map to feature names if lengths match
        if imp_arr and len(imp_arr) == len(feature_columns):
            feature_importance_out[model_name] = dict(zip(feature_columns, imp_arr))
        else:
            # fallback: index-based naming
            feature_importance_out[model_name] = {f"feature_{i}": float(v) for i, v in enumerate(imp_arr)}

    # Model performance
    model_perf = adv.model_performance or {}

    training_stats = {
        'n_samples': len(df),
        'n_features': len(feature_columns),
        'models_trained': list(trainer.models.keys())
    }

    exported = exporter.export_comprehensive_dashboard_data(
        predictions_list, model_perf, feature_importance_out, training_stats
    )

    exporter.create_powerbi_measures_json()
    exporter.generate_powerbi_report_instructions()

    # Save a short training summary
    summary = {
        'results': {k: (v.get('cv_mean') if isinstance(v, dict) else None) for k, v in results.items()},
        'exported_files': exported
    }
    Path('results').mkdir(parents=True, exist_ok=True)
    with open('results/training_summary.json', 'w', encoding='utf-8') as sf:
        json.dump(summary, sf, indent=2)

    logger.info("Training and export complete. See data/powerbi/ for exported files and models/ for artifacts.")


if __name__ == '__main__':
    main()
