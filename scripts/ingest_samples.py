"""
Ingest AI and Human code samples from local folders, build a labeled features CSV,
and optionally retrain models.

Folder layout (recommended):

data/
  train/
    ai/      # AI-generated code files (any supported extension)
    human/   # Human-written code files
  validation/ (optional)
    ai/
    human/

Usage:
  python scripts/ingest_samples.py --split train --out data/processed/features.csv --retrain

Notes:
- Language is auto-detected per file to improve feature extraction context.
- The output CSV includes a 'label' column: 1=AI, 0=Human.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / 'src') not in sys.path:
    sys.path.append(str(ROOT / 'src'))

from preprocessing.feature_extractor import StatisticalFeatureExtractor
from preprocessing.language_detector import LanguageDetector
from preprocessing.ast_parser import ASTFeatureExtractor
from preprocessing.code_tokenizer import AdvancedCodeTokenizer
from utils.data_utils import CodePreprocessor


SUPPORTED_EXTS = {
    '.py', '.java', '.js', '.ts', '.jsx', '.tsx', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.cs', '.go', '.rs'
}


def find_code_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() in SUPPORTED_EXTS:
                files.append(p)
    return files


def ingest_split(split_dir: Path, label_value: int) -> pd.DataFrame:
    extractor = StatisticalFeatureExtractor()
    ast_extractor = ASTFeatureExtractor()
    tokenizer = AdvancedCodeTokenizer()
    lang_det = LanguageDetector()

    rows: List[Dict] = []

    for file_path in find_code_files(split_dir):
        try:
            code_text = file_path.read_text(encoding='utf-8', errors='ignore')
            clean = CodePreprocessor.clean_code(code_text)
            language, lang_conf = lang_det.detect_language(clean, filename=str(file_path.name))
            # Statistical features
            feats = extractor.extract_features(clean, language=language)
            # AST features
            try:
                feats.update(ast_extractor.extract_features(clean, language))
            except Exception:
                pass
            # Token metrics
            try:
                feats.update(tokenizer.get_code_metrics(clean, language))
            except Exception:
                pass
            # Embedding features disabled to align with current training/inference

            # Make robust: replace inf/nan
            for k, v in list(feats.items()):
                if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                    feats[k] = 0.0

            row = {
                'filepath': str(file_path),
                'filename': file_path.name,
                'language': language,
                'language_confidence': lang_conf,
                'label': int(label_value),  # 1=AI, 0=Human
            }
            row.update(feats)
            rows.append(row)

        except Exception:
            # Skip problematic files; keep ingestion resilient
            continue

    return pd.DataFrame(rows)


def build_dataset(data_root: Path, split: str) -> pd.DataFrame:
    split_root = data_root / split
    ai_dir = split_root / 'ai'
    human_dir = split_root / 'human'

    ai_df = ingest_split(ai_dir, label_value=1) if ai_dir.exists() else pd.DataFrame()
    human_df = ingest_split(human_dir, label_value=0) if human_dir.exists() else pd.DataFrame()

    if ai_df.empty and human_df.empty:
        raise FileNotFoundError(f"No files found in {ai_dir} or {human_dir}")

    df = pd.concat([ai_df, human_df], ignore_index=True)
    # Basic cleanup
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='data', help='Root data directory')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'validation', 'test'])
    parser.add_argument('--out', type=str, default='data/processed/features.csv')
    parser.add_argument('--append', action='store_true', help='Append to existing CSV if present')
    parser.add_argument('--retrain', action='store_true', help='Retrain models after writing CSV')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset(data_root, args.split)

    if args.append and out_csv.exists():
        existing = pd.read_csv(out_csv)
        df = pd.concat([existing, df], ignore_index=True)

    # Ensure label is int {0,1} with robust coercion
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        if 'filepath' in df.columns:
            # Infer labels from filepath for missing values
            mask_nan = df['label'].isna()
            if mask_nan.any():
                fp = df.loc[mask_nan, 'filepath'].astype(str)
                inferred = fp.str.contains(r"[\\/]ai[\\/]")
                df.loc[mask_nan, 'label'] = inferred.astype(int)
        # Any remaining NaNs -> 0 (conservative)
        df['label'] = df['label'].fillna(0).astype(int)

    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")

    if args.retrain:
        # Invoke the training script as a subprocess to avoid import path issues
        import subprocess, sys as _sys
        print("Starting retraining...")
        cmd = [
            _sys.executable,
            str(Path(__file__).resolve().parent / 'train_and_export.py')
        ]
        subprocess.check_call(cmd, cwd=str(Path(__file__).resolve().parents[1]))


if __name__ == '__main__':
    main()


