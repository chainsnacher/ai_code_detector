#!/usr/bin/env python3
"""
Comprehensive test to verify AI vs Human code differentiation
"""
import sys
sys.path.insert(0, '.')
from src.preprocessing.feature_extractor import StatisticalFeatureExtractor
from src.preprocessing.code_tokenizer import AdvancedCodeTokenizer
from src.preprocessing.ast_parser import ASTFeatureExtractor
from src.models.baseline_models import BaselineModelTrainer
import json
import pandas as pd
import numpy as np

# Load models once globally
_baseline_trainer = None

def get_baseline_trainer():
    global _baseline_trainer
    if _baseline_trainer is None:
        _baseline_trainer = BaselineModelTrainer()
        _baseline_trainer.load_models('models/baseline')
    return _baseline_trainer

# Multiple test samples
test_samples = {
    "HUMAN - Simple Function": ("""def add(a, b):
    return a + b
""", 0),
    
    "HUMAN - With Comments": ("""def calculate_total(items):
    # Sum all items
    total = 0
    for item in items:
        total += item
    return total
""", 0),
    
    "HUMAN - Practical Code": ("""def process_data(data_list):
    result = []
    for item in data_list:
        if item > 0:
            result.append(item * 2)
    return result
""", 0),
    
    "HUMAN - Real Pattern": ("""class DataProcessor:
    def __init__(self, name):
        self.name = name
        self.data = []
    
    def add(self, value):
        self.data.append(value)
    
    def get_sum(self):
        return sum(self.data)
""", 0),
    
    "AI - Verbose Function": ("""def compute_sum_of_numeric_values(numeric_list):
    \"\"\"
    Calculate the sum of all numeric values in a list.
    This function iterates through the provided list and computes the cumulative sum.
    
    Args:
        numeric_list: A list of numeric values to be summed
        
    Returns:
        The total sum of all values in the list
    \"\"\"
    total_sum = 0
    for numeric_value in numeric_list:
        total_sum = total_sum + numeric_value
    return total_sum
""", 1),
    
    "AI - Over-documented": ("""def calculate_maximum_value_from_list(input_list):
    \"\"\"
    This function determines and returns the maximum value present in the input list.
    It uses Python's built-in max function for efficient computation.
    \"\"\"
    # Check if list is not empty
    if len(input_list) > 0:
        maximum_value = max(input_list)
        return maximum_value
    else:
        return None
""", 1),
    
    "AI - Template-like": ("""def extract_features_from_data(input_data):
    processed_data = []
    for element in input_data:
        if element is not None:
            processed_data.append(element)
    normalized_data = [x / sum(processed_data) for x in processed_data]
    return normalized_data

def create_model_ensemble():
    ensemble_models = []
    ensemble_models.append('model_one')
    ensemble_models.append('model_two')
    ensemble_models.append('model_three')
    return ensemble_models
""", 1),

    "AI - Verbose Naming": ("""def validate_user_input_and_process_accordingly(user_provided_input_data):
    validated_and_processed_output_result = []
    for individual_input_element in user_provided_input_data:
        if individual_input_element is not None and individual_input_element != '':
            processed_element = individual_input_element.strip().lower()
            validated_and_processed_output_result.append(processed_element)
    return validated_and_processed_output_result
""", 1),
}

def get_predictions(code):
    """Extract features and get model predictions"""
    ast_extractor = ASTFeatureExtractor()
    stat_extractor = StatisticalFeatureExtractor()
    tokenizer = AdvancedCodeTokenizer()
    
    features_dict = {}
    try:
        features_dict.update(ast_extractor.extract_features(code, 'python'))
    except:
        pass
    try:
        features_dict.update(stat_extractor.extract_features(code, 'python'))
    except:
        pass
    try:
        features_dict.update(tokenizer.get_code_metrics(code, 'python'))
    except:
        pass
    
    if not features_dict:
        return None, None, None, None
    
    # Load canonical columns
    try:
        with open('models/feature_columns.json', 'r') as f:
            canonical_cols = json.load(f)
    except:
        return None, None, None, None
    
    # Align features
    try:
        features_df = pd.DataFrame([features_dict])
        features_df = features_df.reindex(columns=canonical_cols, fill_value=0)
        X = features_df.values.astype(float)
        X = np.nan_to_num(X, copy=False)
    except:
        return None, None, None, None
    
    # Get predictions from ensemble
    try:
        baseline = get_baseline_trainer()
    except:
        return None, None, None, None
    
    ensemble_pred = None
    ensemble_prob = None
    human_prob = None
    ai_prob = None
    
    if 'ensemble' in baseline.models:
        model = baseline.models['ensemble']
        try:
            pred = model.predict(X)
            ensemble_pred = int(pred[0])
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                human_prob = float(probs[0][0])
                ai_prob = float(probs[0][1])
                ensemble_prob = max(human_prob, ai_prob)
        except:
            pass
    
    return ensemble_pred, ensemble_prob, human_prob, ai_prob

# Run tests
print("\n" + "="*80)
print("COMPREHENSIVE AI vs HUMAN CODE DIFFERENTIATION TEST")
print("="*80)

correct_predictions = 0
total_tests = len(test_samples)

results = []

for label, (code, expected) in test_samples.items():
    pred, conf, human_p, ai_p = get_predictions(code)
    
    if pred is None:
        result_str = "ERROR"
        is_correct = False
    else:
        is_correct = (pred == expected)
        correct_predictions += is_correct if is_correct else 0
        result_str = "✓ CORRECT" if is_correct else "✗ WRONG"
    
    expected_str = "👤 HUMAN" if expected == 0 else "🤖 AI"
    pred_str = "👤 HUMAN" if pred == 0 else "🤖 AI" if pred == 1 else "ERROR"
    
    print(f"\n{label}")
    print(f"  Expected: {expected_str}")
    print(f"  Predicted: {pred_str}")
    if conf is not None:
        print(f"  Confidence: {conf:.1%}")
        print(f"  Human Prob: {human_p:.1%} | AI Prob: {ai_p:.1%}")
    print(f"  {result_str}")
    
    results.append({
        'label': label,
        'expected': expected,
        'predicted': pred,
        'confidence': conf,
        'correct': is_correct
    })

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
accuracy = (correct_predictions / total_tests) * 100
print(f"Correct Predictions: {correct_predictions}/{total_tests}")
print(f"Accuracy: {accuracy:.1f}%")

if accuracy >= 80:
    print("✅ SYSTEM IS PROPERLY DIFFERENTIATING AI vs HUMAN CODE!")
elif accuracy >= 60:
    print("⚠️  SYSTEM IS WORKING BUT COULD BE IMPROVED")
else:
    print("❌ SYSTEM NEEDS IMPROVEMENT")

print("="*80)
