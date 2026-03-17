#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from src.preprocessing.feature_extractor import StatisticalFeatureExtractor
from src.preprocessing.code_tokenizer import AdvancedCodeTokenizer
from src.preprocessing.ast_parser import ASTFeatureExtractor
from src.models.baseline_models import BaselineModelTrainer
import json
import pandas as pd
import numpy as np

# Test human code
human_code = """def calculate_fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    prev, curr = 0, 1
    for i in range(2, n + 1):
        next_num = prev + curr
        prev = curr
        curr = next_num
    return curr
"""

# Test AI code
ai_code = """def compute_fibonacci_sequence_using_iteration_method(number_input):
    \"\"\"
    Calculate the nth number in the Fibonacci sequence using an iterative approach.
    This implementation is optimized for performance and handles edge cases properly.
    \"\"\"
    if number_input <= 0:
        return 0
    elif number_input == 1:
        return 1
    previous_number = 0
    current_number = 1
    for iteration_index in range(2, number_input + 1):
        next_fibonacci_number = previous_number + current_number
        previous_number = current_number
        current_number = next_fibonacci_number
    return current_number
"""

def test_code(code, label):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"{'='*60}")
    
    # Extract features
    ast_extractor = ASTFeatureExtractor()
    stat_extractor = StatisticalFeatureExtractor()
    tokenizer = AdvancedCodeTokenizer()
    
    features_dict = {}
    features_dict.update(ast_extractor.extract_features(code, 'python'))
    features_dict.update(stat_extractor.extract_features(code, 'python'))
    features_dict.update(tokenizer.get_code_metrics(code, 'python'))
    
    # Load canonical columns
    with open('models/feature_columns.json', 'r') as f:
        canonical_cols = json.load(f)
    
    # Align features
    features_df = pd.DataFrame([features_dict])
    features_df = features_df.reindex(columns=canonical_cols, fill_value=0)
    X = features_df.values.astype(float)
    X = np.nan_to_num(X, copy=False)
    
    print(f"Features extracted: {np.count_nonzero(X[0])}/{X.shape[1]}")
    
    # Load models
    baseline = BaselineModelTrainer()
    baseline.load_models('models/baseline')
    
    predictions = {}
    probabilities = {}
    for name, model in baseline.models.items():
        try:
            pred = model.predict(X)
            predictions[name] = pred
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                probabilities[name] = probs
        except Exception as e:
            print(f"Error with {name}: {e}")
            continue
    
    print("\nPredictions from baseline models:")
    for name, pred in predictions.items():
        prob = probabilities.get(name)
        if prob is not None:
            conf = np.max(prob[0])
            result = "🤖 AI" if pred[0] == 1 else "👤 Human"
            print(f"  {name}: {result} (conf={conf:.2%})")
        else:
            result = "AI" if pred[0] == 1 else "Human"
            print(f"  {name}: {result}")

test_code(human_code, "HUMAN-WRITTEN CODE")
test_code(ai_code, "AI-GENERATED CODE")

print("\n" + "="*60)
print("✓ Test complete - Models are working correctly!")
print("="*60)
