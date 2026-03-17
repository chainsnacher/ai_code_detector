#!/usr/bin/env python3
"""
Complete retraining pipeline for AI Code Detector.

This script:
1. Collects/generates real training data (AI and human code)
2. Extracts comprehensive features
3. Trains models with proper class balancing
4. Evaluates performance
5. Saves models for production use

Usage:
    python retrain_full_pipeline.py [--collect-data] [--skip-data-collection]
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def check_dependencies():
    """Check if required dependencies are installed."""
    # Map package names to their import names
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',  # scikit-learn imports as sklearn
        'joblib': 'joblib'
    }
    
    missing = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.info("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def collect_training_data():
    """Collect real training data."""
    logger.info("="*60)
    logger.info("STEP 1: Collecting Training Data")
    logger.info("="*60)
    
    collect_script = ROOT / 'scripts' / 'collect_real_data.py'
    
    if not collect_script.exists():
        logger.error(f"Data collection script not found: {collect_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(collect_script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Data collection failed:\n{result.stderr}")
            return False
        
        logger.info("Data collection completed successfully")
        logger.info(result.stdout)
        return True
    
    except Exception as e:
        logger.error(f"Error running data collection: {e}")
        return False


def train_models():
    """Train models using improved pipeline."""
    logger.info("="*60)
    logger.info("STEP 2: Training Models")
    logger.info("="*60)
    
    train_script = ROOT / 'scripts' / 'improved_train_pipeline.py'
    
    if not train_script.exists():
        logger.error(f"Training script not found: {train_script}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Training failed:\n{result.stderr}")
            logger.info("STDOUT:", result.stdout)
            return False
        
        logger.info("Training completed successfully")
        logger.info(result.stdout)
        return True
    
    except Exception as e:
        logger.error(f"Error running training: {e}")
        return False


def verify_models():
    """Verify that models were saved correctly."""
    logger.info("="*60)
    logger.info("STEP 3: Verifying Models")
    logger.info("="*60)
    
    models_dir = ROOT / 'models'
    baseline_dir = models_dir / 'baseline'
    ensemble_dir = models_dir / 'ensemble'
    feature_cols_file = models_dir / 'feature_columns.json'
    
    checks = {
        'Models directory exists': models_dir.exists(),
        'Baseline models directory exists': baseline_dir.exists(),
        'Ensemble directory exists': ensemble_dir.exists(),
        'Feature columns file exists': feature_cols_file.exists(),
    }
    
    if baseline_dir.exists():
        model_files = list(baseline_dir.glob('*_model.pkl'))
        checks[f'Baseline model files ({len(model_files)} found)'] = len(model_files) > 0
    
    if ensemble_dir.exists():
        ensemble_files = list(ensemble_dir.glob('*.pkl'))
        checks[f'Ensemble files ({len(ensemble_files)} found)'] = len(ensemble_files) > 0
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        logger.info(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\nAll model verification checks passed!")
    else:
        logger.warning("\nSome verification checks failed. Models may not be ready for use.")
    
    return all_passed


def print_summary():
    """Print training summary and next steps."""
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    models_dir = ROOT / 'models'
    results_file = models_dir / 'training_results.json'
    
    if results_file.exists():
        import json
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            logger.info("\nTest Set Performance:")
            test_results = results.get('test_results', {})
            for model_name, metrics in sorted(
                test_results.items(),
                key=lambda x: x[1].get('f1_score', 0),
                reverse=True
            ):
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1_score', 0)
                auc = metrics.get('roc_auc', 0)
                logger.info(f"  {model_name:20s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}" if auc else f"  {model_name:20s} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        except Exception as e:
            logger.debug(f"Could not read results file: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS")
    logger.info("="*60)
    logger.info("1. Test the models:")
    logger.info("   python test_model_predictions.py")
    logger.info("\n2. Run the web app:")
    logger.info("   streamlit run web_app/app.py")
    logger.info("\n3. For batch predictions:")
    logger.info("   python quick_test.py")


def main():
    """Main retraining function."""
    parser = argparse.ArgumentParser(
        description='Retrain the full AI Code Detector pipeline'
    )
    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Collect new training data before training'
    )
    parser.add_argument(
        '--skip-data-collection',
        action='store_true',
        help='Skip data collection (use existing data)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training (only verify existing models)'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("AI CODE DETECTOR - FULL PIPELINE RETRAINING")
    logger.info("="*60)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Please install missing dependencies and try again.")
        return 1
    
    # Step 1: Collect data (if requested or if no data exists)
    data_dir = ROOT / 'data' / 'train'
    human_dir = data_dir / 'human'
    ai_dir = data_dir / 'ai'
    
    should_collect = args.collect_data or (
        not args.skip_data_collection and
        (not human_dir.exists() or not ai_dir.exists() or
         len(list(human_dir.glob('*'))) == 0 or len(list(ai_dir.glob('*'))) == 0)
    )
    
    if should_collect:
        if not collect_training_data():
            logger.error("Data collection failed. Cannot proceed with training.")
            return 1
    else:
        logger.info("Skipping data collection (using existing data)")
    
    # Step 2: Train models
    if not args.skip_training:
        if not train_models():
            logger.error("Training failed. Check logs for details.")
            return 1
    else:
        logger.info("Skipping training (using existing models)")
    
    # Step 3: Verify models
    if not verify_models():
        logger.warning("Model verification found issues. Check logs above.")
        return 1
    
    # Step 4: Print summary
    print_summary()
    
    logger.info("\n" + "="*60)
    logger.info("RETRAINING COMPLETE!")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
