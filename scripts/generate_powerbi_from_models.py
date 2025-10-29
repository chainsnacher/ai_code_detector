"""
Load existing baseline models and produce Power BI CSV exports (predictions, feature importance, measures).
This is a fallback when no labelled dataset exists — it does inference with existing models and creates dashboard CSVs for demo.
"""
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from src.utils.data_utils import DataProcessor
from src.models.baseline_models import BaselineModelTrainer
from src.utils.powerbi_exporter import PowerBIExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_code_detector")

FEATURES_CSV = Path("data/processed/features.csv")
FEATURE_COLUMNS_FILE = Path("models/feature_columns.json")
BASELINE_DIR = Path("models/baseline")


def main():
    logger.info("Starting generate_powerbi_from_models")

    if not FEATURES_CSV.exists():
        logger.error("Features CSV not found; aborting")
        return

    if not BASELINE_DIR.exists():
        logger.error("Baseline models not found; aborting")
        return

    df = pd.read_csv(FEATURES_CSV)
    if FEATURE_COLUMNS_FILE.exists():
        feature_columns = json.load(open(FEATURE_COLUMNS_FILE, 'r'))
    else:
        # fallback to all columns in df
        feature_columns = list(df.columns)

    # Load baseline models
    trainer = BaselineModelTrainer()
    trainer.load_models(str(BASELINE_DIR))

    # Prepare X
    X = df[feature_columns].values

    # Predictions per model
    predictions_list = []
    n = len(df)

    # Collect per-model outputs
    per_model_preds = {}
    per_model_probs = {}

    for model_name, model in trainer.models.items():
        try:
            preds = model.predict(X)
            per_model_preds[model_name] = preds
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                per_model_probs[model_name] = probs
            else:
                per_model_probs[model_name] = None
        except Exception as e:
            logger.warning(f"Error predicting with {model_name}: {e}")

    # Build final prediction via simple majority vote across available preds
    import numpy as np

    if per_model_preds:
        pred_array = np.array(list(per_model_preds.values()))
        majority_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=pred_array)
    else:
        majority_preds = [0] * n

    # Calculate a simple confidence as mean of positive-class probabilities when available
    avg_confidence = [0.0] * n
    if per_model_probs:
        confidences = []
        for probs in per_model_probs.values():
            if probs is None:
                continue
            if probs.shape[1] == 2:
                confidences.append(probs[:, 1])
            else:
                # fallback: max prob
                confidences.append(np.max(probs, axis=1))
        if confidences:
            avg_confidence = np.mean(np.vstack(confidences), axis=0)
        else:
            avg_confidence = [0.0] * n

    for i in range(n):
        predictions_list.append({
            'id': i,
            'code_hash': '',
            'language': '',
            'prediction': int(majority_preds[i]),
            'confidence': float(avg_confidence[i]) if hasattr(avg_confidence[i], '__float__') else float(avg_confidence[i]),
            'model_name': 'baseline_majority_vote',
            'timestamp': datetime.now().isoformat(),
            'code_sample': ''
        })

    # Feature importance mapping
    feature_importance = {}
    # If trainer.feature_importance is empty, try loading file
    if not trainer.feature_importance:
        imp_file = BASELINE_DIR / "feature_importance.pkl"
        if imp_file.exists():
            import joblib
            trainer.feature_importance = joblib.load(imp_file)

    for model_name, imp in trainer.feature_importance.items():
        try:
            imp_list = list(imp)
        except Exception:
            imp_list = []
        if imp_list and len(imp_list) == len(feature_columns):
            feature_importance[model_name] = dict(zip(feature_columns, imp_list))
        else:
            feature_importance[model_name] = {f"f_{i}": float(v) for i, v in enumerate(imp_list)}

    exporter = PowerBIExporter()
    exported = exporter.export_comprehensive_dashboard_data(predictions_list, {}, feature_importance, {'n_samples': n})
    exporter.create_powerbi_measures_json()
    exporter.generate_powerbi_report_instructions()

    logger.info(f"Exported Power BI files: {exported}")


if __name__ == '__main__':
    main()
