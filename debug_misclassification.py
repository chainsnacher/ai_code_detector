import sys
import os
import pandas as pd
import numpy as np
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing.feature_extractor import StatisticalFeatureExtractor
from src.preprocessing.ast_parser import ASTFeatureExtractor
from src.preprocessing.code_tokenizer import AdvancedCodeTokenizer
from src.models.baseline_models import BaselineModelTrainer

def load_models():
    try:
        baseline_trainer = BaselineModelTrainer()
        baseline_trainer.load_models('models/baseline')
        return baseline_trainer
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def extract_features(code):
    features = {}
    try:
        ast_extractor = ASTFeatureExtractor()
        features.update(ast_extractor.extract_features(code, 'python'))
    except: pass
    try:
        stat_extractor = StatisticalFeatureExtractor()
        features.update(stat_extractor.extract_features(code, 'python'))
    except: pass
    try:
        tokenizer = AdvancedCodeTokenizer()
        features.update(tokenizer.get_code_metrics(code, 'python'))
    except: pass
    return features

def main():
    print("Loading models...")
    trainer = load_models()
    if not trainer: return

    with open('models/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)

    # Reconstruct code from screenshot
    user_code = """
try:
    num1 = int(input("Enter first integer: "))
    num2 = int(input("Enter second integer: "))

    result = lcm(num1, num2)
    print(f"LCM of {num1} and {num2} is {result}")

except ValueError:
    print("Invalid input. Please enter integers only.")

if __name__ == "__main__":
    main()
"""

    print("Extracting features...")
    feats = extract_features(user_code)
    
    # Vectorize
    feat_vector = []
    for col in feature_columns:
        feat_vector.append(feats.get(col, 0))
    X = np.array([feat_vector])

    print("\n--- Predictions ---")
    for name, model in trainer.models.items():
        try:
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            print(f"Model: {name}")
            print(f"  Prediction: {pred} (0=Human, 1=AI)")
            print(f"  Probability: {prob}")
        except Exception as e:
            print(f"  {name} failed: {e}")

    print("\n--- Key Features ---")
    print(f"Shannon Entropy: {feats.get('shannon_entropy', 0)}")
    print(f"Compression Ratio: {feats.get('compression_ratio', 0)}")
    print(f"Line Length Variance: {feats.get('line_length_variance', 0)}")
    print(f"Docstring Ratio: {feats.get('docstring_ratio', 0)}")
    print(f"Comment Ratio: {feats.get('comment_ratio', 0)}")

if __name__ == "__main__":
    main()
