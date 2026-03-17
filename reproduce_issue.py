import sys
import os
import pandas as pd
import joblib
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.preprocessing.feature_extractor import StatisticalFeatureExtractor
from src.preprocessing.ast_parser import ASTFeatureExtractor
from src.preprocessing.code_tokenizer import AdvancedCodeTokenizer

from src.models.baseline_models import BaselineModelTrainer

def load_models():
    """Load the trained ensemble model and base models."""
    try:
        # Load baseline models
        baseline_trainer = BaselineModelTrainer()
        baseline_trainer.load_models('models/baseline')
        
        # Load ensemble
        from src.models.ensemble_model import AdvancedEnsembleDetector
        ensemble = AdvancedEnsembleDetector()
        ensemble.load_ensemble('models/ensemble')
        
        # Add baseline models to ensemble
        for name, model in baseline_trainer.models.items():
            ensemble.add_base_model(f'baseline_{name}', model)
            
        return ensemble
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_features(code):
    """Extract features for a code snippet."""
    features = {}
    
    # AST
    try:
        ast_extractor = ASTFeatureExtractor()
        features.update(ast_extractor.extract_features(code, 'python'))
    except: pass

    # Statistical
    try:
        stat_extractor = StatisticalFeatureExtractor()
        features.update(stat_extractor.extract_features(code, 'python'))
    except: pass
    
    # Token
    try:
        tokenizer = AdvancedCodeTokenizer()
        features.update(tokenizer.get_code_metrics(code, 'python'))
    except: pass
    
    return features

def main():
    print("Loading model...")
    model = load_models()
    if not model:
        print("Could not load model. Ensure 'models/ensemble' exists.")
        return

    # Load feature columns (needed to align features)
    try:
        import json
        with open('models/feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
    except:
        print("Could not load feature_columns.json")
        return

    # Defines test cases
    ai_code = """
def calculate_factorial(n):
    \"\"\"Calculates the factorial of a non-negative integer n.\"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
"""
    
    human_code = """
def fact(n):
    # simple factorial
    if n<=1: return 1
    return n * fact(n-1)
"""

    user_sample_ai = """
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

    samples = [
        ("AI Code (Standard)", ai_code, 1),
        ("Human Code (Standard)", human_code, 0),
        ("User Sample (AI)", user_sample_ai, 1)
    ]

    print("\nRunning Predictions:", flush=True)
    print("-" * 30, flush=True)

    for name, code, label in samples:
        try:
            # Extract
            feats = extract_features(code)
            
            # Align to training columns
            feat_vector = []
            for col in feature_columns:
                feat_vector.append(feats.get(col, 0))
            
            print(f"DEBUG: Vector length: {len(feat_vector)}, Expected columns: {len(feature_columns)}", flush=True)
            # Predict
            X = np.array([feat_vector])
            
            # The ensemble predict expects a 2D array
            pred = model.predict(X)[0]
            
            result_str = "AI" if pred == 1 else "Human"
            correct = (pred == label)
            
            print(f"Sample: {name}", flush=True)
            print(f"Prediction: {result_str} (Expected: {'AI' if label == 1 else 'Human'})", flush=True)
            print(f"Correct: {correct}", flush=True)
            print("-" * 30, flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error processing sample {name}: {e}", flush=True)

if __name__ == "__main__":
    main()
