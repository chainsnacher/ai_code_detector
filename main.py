"""
Main training pipeline for the AI Code Detection System.
Orchestrates the complete training process from data collection to model deployment.
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import json
import hashlib
import random
from datetime import datetime

# Add src to path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.utils.config import get_config
from src.utils.logger import setup_logger, get_performance_logger
from src.utils.database import get_database
from src.utils.data_utils import DataProcessor, DataSaver, DataBalancer
# from utils.data_labeler import DataLabeler  # Temporarily disabled
from src.utils.powerbi_exporter import PowerBIExporter
from src.preprocessing.language_detector import LanguageDetector

# Feature extraction
from src.preprocessing.ast_parser import ASTFeatureExtractor
from src.preprocessing.feature_extractor import StatisticalFeatureExtractor
# from src.preprocessing.embedding_generator import CodeEmbeddingGenerator, EmbeddingEnsemble  # Disabled
from src.preprocessing.code_tokenizer import AdvancedCodeTokenizer

# Models
from src.models.baseline_models import BaselineModelTrainer, AdvancedEnsemble
# from src.models.transformer_model import TransformerTrainer, MultiModelEnsemble  # Disabled
from src.models.ensemble_model import AdvancedEnsembleDetector

# Evaluation
from src.evaluation.metrics import AdvancedMetrics
from src.evaluation.cross_validation import AdvancedCrossValidator
from src.evaluation.adversarial_testing import AdversarialTester

logger = setup_logger()
perf_logger = get_performance_logger()
config = get_config()
db = get_database()

class AICodeDetectionPipeline:
    """Main pipeline for AI code detection system."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.config = config
        self.logger = logger
        self.perf_logger = perf_logger
        self.db = db
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.data_saver = DataSaver()
        self.data_balancer = DataBalancer()
        # self.data_labeler = DataLabeler()  # Temporarily disabled
        self.powerbi_exporter = PowerBIExporter()
        self.language_detector = LanguageDetector()
        
        # Feature extractors
        self.ast_extractor = ASTFeatureExtractor()
        self.statistical_extractor = StatisticalFeatureExtractor()
        # self.embedding_generator = CodeEmbeddingGenerator(config.get_section('features'))  # Disabled to avoid network calls
        self.embedding_generator = None
        self.tokenizer = AdvancedCodeTokenizer()
        
        # Models
        self.baseline_trainer = BaselineModelTrainer(config.get_section('models'))
        self.transformer_trainer = None
        self.ensemble_detector = AdvancedEnsembleDetector(config.get_section('models'))
        
        # Evaluation
        self.metrics_calculator = AdvancedMetrics()
        self.cross_validator = AdvancedCrossValidator(config.get_section('evaluation'))
        self.adversarial_tester = AdversarialTester(config.get_section('evaluation'))
        
        # Training data
        self.training_data = None
        self.features = None
        self.labels = None
        
        # Results
        self.training_results = {}
        self.evaluation_results = {}
    
    def run_complete_pipeline(self, skip_data=False, skip_training=False, skip_evaluation=False, skip_adversarial=False):
        """Run the complete training and evaluation pipeline."""
        self.logger.info("Starting AI Code Detection Pipeline")
        start_time = time.time()
        
        try:
            # Step 1: Data Collection and Preparation
            if not skip_data:
                self.logger.info("Step 1: Data Collection and Preparation")
                self._collect_and_prepare_data()
            else:
                self.logger.info("Step 1: Skipping data collection")
            
            # Step 2: Feature Engineering
            self.logger.info("Step 2: Feature Engineering")
            self._extract_features()
            
            # Step 3: Model Training
            if not skip_training:
                self.logger.info("Step 3: Model Training")
                self._train_models()
            else:
                self.logger.info("Step 3: Skipping model training")
            
            # Step 4: Model Evaluation
            if not skip_evaluation:
                self.logger.info("Step 4: Model Evaluation")
                self._evaluate_models()
            else:
                self.logger.info("Step 4: Skipping evaluation")
            
            # Step 5: Ensemble Creation
            self.logger.info("Step 5: Ensemble Creation")
            self._create_ensemble()
            
            # Step 6: Adversarial Testing
            if not skip_adversarial:
                self.logger.info("Step 6: Adversarial Testing")
                self._test_robustness()
            else:
                self.logger.info("Step 6: Skipping adversarial testing")
            
            # Step 7: Save Results
            self.logger.info("Step 7: Save Results")
            self._save_results()
            
            total_time = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _collect_and_prepare_data(self):
        """Collect and prepare training data."""
        self.logger.info("Creating synthetic training data")
        
        # Generate synthetic code samples (more data + variety for better human vs AI detection)
        human_code_samples = self._generate_human_code_samples(3000)
        ai_code_samples = self._generate_ai_code_samples(3000)
        
        # Combine and label data
        all_samples = human_code_samples + ai_code_samples
        labels = [0] * len(human_code_samples) + [1] * len(ai_code_samples)
        languages = ['python'] * len(all_samples)
        
        # Create DataFrame
        self.training_data = pd.DataFrame({
            'code': all_samples,
            'label': labels,
            'language': languages
        })
        
        # Balance dataset if needed
        if config.get('data_balancing', {}).get('enabled', False):
            self.training_data = self.data_balancer.balance_dataset(
                self.training_data, 'label', method='undersample'
            )
        
        self.logger.info(f"Created dataset with {len(self.training_data)} samples")
        
        # Save to database (commented out to avoid connection issues)
        # for idx, row in self.training_data.iterrows():
        #     self.db.save_code_sample(
        #         code_hash=f"sample_{idx}",
        #         code_sample=row['code'],
        #         language=row['language'],
        #         label=row['label'],
        #         source='synthetic'
        #     )
    
    def _generate_human_code_samples(self, n_samples: int) -> List[str]:
        """Generate diverse human-like code: terse, varied style, short/medium/long."""
        samples = []
        random.seed(42)
        
        # Short / one-liner style
        short_human = [
            "x = 1\nprint(x)",
            "a = [1,2,3]\nprint(min(a))",
            "def f(): return 42",
            "for i in range(3): print(i*i)",
            "s = 'hi'\nprint(s.upper())",
            "nums = [32,54,67]\nsmallest = min(nums)\nprint(smallest)",
            "if x > 0: print('ok')",
            "def add(a,b): return a+b",
            "lst = [1,2,3]\nprint(lst[-1])",
            "d = {'a':1}\nprint(d.get('a',0))",
            "r = sum([1,2,3])",
            "def double(n):\n  return n*2",
            "out = [x*2 for x in range(5)]",
            "import sys\nprint(sys.version)",
            "try:\n  x=1\nexcept: pass",
        ]
        # Medium – terse names, minimal comments
        medium_human = [
            """
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
""",
            """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
""",
            """
def bsearch(arr, t):
    lo, hi = 0, len(arr)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if arr[mid] == t: return mid
        if arr[mid] < t: lo = mid+1
        else: hi = mid-1
    return -1
""",
            """
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    def append(self, data):
        n = Node(data)
        if not self.head:
            self.head = n
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = n
""",
            """
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True
""",
            """
def qsort(arr):
    if len(arr) <= 1: return arr
    p = arr[len(arr)//2]
    L = [x for x in arr if x < p]
    M = [x for x in arr if x == p]
    R = [x for x in arr if x > p]
    return qsort(L) + M + qsort(R)
""",
            """
def count_words(s):
    c = {}
    for w in s.split():
        c[w] = c.get(w,0) + 1
    return c
""",
            """
def greet(user):
    return f"Hi, {user}"
print(greet("you"))
""",
        ]
        # Longer – still human style (mixed indentation, fewer docstrings)
        long_human = [
            """
def process_data(data):
    out = []
    for item in data:
        if item and len(item) > 0:
            out.append(item.strip())
    return out

def main():
    d = ["a ", " b", "c"]
    r = process_data(d)
    print(r)
main()
""",
            """
class Stack:
    def __init__(self):
        self._d = []
    def push(self, x):
        self._d.append(x)
    def pop(self):
        return self._d.pop() if self._d else None
    def __len__(self):
        return len(self._d)
""",
        ]
        all_human = short_human + medium_human + long_human
        
        for i in range(n_samples):
            t = all_human[i % len(all_human)]
            # Name variations only (avoid changing indentation to prevent syntax errors)
            if i % 4 == 0 and 'def ' in t:
                t = t.replace('def ', 'def _', 1)
            samples.append(t.strip())
        
        return samples
    
    def _generate_ai_code_samples(self, n_samples: int) -> List[str]:
        """Generate AI-like code: docstrings, verbose names, consistent style."""
        samples = []
        random.seed(43)
        
        # Short but AI-style (verbose names / comments)
        short_ai = [
            "user_input_value = 1\nprint(user_input_value)",
            "list_of_numbers = [1, 2, 3]\nprint(min(list_of_numbers))",
            "def get_default_value():\n    return 42",
            "for index in range(3):\n    print(index * index)",
            "string_value = 'hello'\nprint(string_value.upper())",
            "numbers_list = [32, 54, 67]\nminimum_value = min(numbers_list)\nprint(minimum_value)",
            "if condition_value > 0:\n    print('success')",
            "def add_two_numbers(first, second):\n    return first + second",
        ]
        # Medium – docstrings, descriptive names
        medium_ai = [
            """
def calculate_fibonacci_sequence(n):
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 1:
        return n
    else:
        return calculate_fibonacci_sequence(n-1) + calculate_fibonacci_sequence(n-2)
""",
            """
def perform_bubble_sort_algorithm(input_array):
    \"\"\"Sort the input array using bubble sort.\"\"\"
    array_length = len(input_array)
    for i in range(array_length):
        for j in range(0, array_length - i - 1):
            if input_array[j] > input_array[j + 1]:
                input_array[j], input_array[j + 1] = input_array[j + 1], input_array[j]
    return input_array
""",
            """
def execute_binary_search_algorithm(sorted_array, target_value):
    \"\"\"Perform binary search on sorted array.\"\"\"
    left_index = 0
    right_index = len(sorted_array) - 1
    while left_index <= right_index:
        middle_index = (left_index + right_index) // 2
        if sorted_array[middle_index] == target_value:
            return middle_index
        elif sorted_array[middle_index] < target_value:
            left_index = middle_index + 1
        else:
            right_index = middle_index - 1
    return -1
""",
            """
class DataNode:
    \"\"\"Node for linked list.\"\"\"
    def __init__(self, data_value):
        self.data_value = data_value
        self.next_node = None

class LinkedListDataStructure:
    \"\"\"A linked list implementation.\"\"\"
    def __init__(self):
        self.head_node = None
    
    def add_element(self, data_value):
        new_node = DataNode(data_value)
        if not self.head_node:
            self.head_node = new_node
            return
        current_node = self.head_node
        while current_node.next_node:
            current_node = current_node.next_node
        current_node.next_node = new_node
""",
            """
def check_if_number_is_prime(number):
    \"\"\"Check whether the number is prime.\"\"\"
    if number < 2:
        return False
    for divisor in range(2, int(number**0.5) + 1):
        if number % divisor == 0:
            return False
    return True
""",
            """
def quicksort_algorithm(input_list):
    \"\"\"Sort list using quicksort.\"\"\"
    if len(input_list) <= 1:
        return input_list
    pivot_element = input_list[len(input_list) // 2]
    left_part = [x for x in input_list if x < pivot_element]
    middle_part = [x for x in input_list if x == pivot_element]
    right_part = [x for x in input_list if x > pivot_element]
    return quicksort_algorithm(left_part) + middle_part + quicksort_algorithm(right_part)
""",
            """
def count_word_frequencies(input_string):
    \"\"\"Count frequency of each word in the string.\"\"\"
    frequency_dictionary = {}
    for word in input_string.split():
        frequency_dictionary[word] = frequency_dictionary.get(word, 0) + 1
    return frequency_dictionary
""",
            """
def generate_greeting_message(username):
    \"\"\"Generate a greeting for the user.\"\"\"
    return f"Welcome, {username}."
print(generate_greeting_message("Pythonista"))
""",
        ]
        # Longer – docstrings, type-hint style, consistent formatting
        long_ai = [
            """
def process_input_data(data_list):
    \"\"\"
    Process the input data list and return cleaned items.
    \"\"\"
    result_list = []
    for item in data_list:
        if item is not None and len(item) > 0:
            result_list.append(item.strip())
    return result_list

def main_execution():
    sample_data = ["a ", " b", "c"]
    processed_result = process_input_data(sample_data)
    print(processed_result)

main_execution()
""",
            """
class StackDataStructure:
    \"\"\"A stack implementation using a list.\"\"\"
    def __init__(self):
        self._internal_list = []
    
    def push_element(self, element):
        self._internal_list.append(element)
    
    def pop_element(self):
        return self._internal_list.pop() if self._internal_list else None
    
    def __len__(self):
        return len(self._internal_list)
""",
            """
def find_minimum_value_in_list(number_list):
    \"\"\"Find and return the minimum value in the given list.\"\"\"
    if not number_list:
        return None
    minimum_value = min(number_list)
    return minimum_value

numbers = [32, 54, 67, 21, -5]
smallest_value = find_minimum_value_in_list(numbers)
print(smallest_value)
""",
        ]
        all_ai = short_ai + medium_ai + long_ai
        
        for i in range(n_samples):
            t = all_ai[i % len(all_ai)]
            if i % 5 == 0 and 'def ' in t:
                t = t.replace('def ', 'def compute_', 1)
            samples.append(t.strip())
        
        return samples
    
    def _extract_features(self):
        """Extract comprehensive features from code samples."""
        self.logger.info("Extracting features from code samples")
        
        all_features = []
        
        for idx, row in self.training_data.iterrows():
            code = row['code']
            language = row['language']
            
            features = {}
            
            # AST features
            try:
                ast_features = self.ast_extractor.extract_features(code, language)
                features.update(ast_features)
            except Exception as e:
                self.logger.warning(f"AST extraction failed for sample {idx}: {e}")
            
            # Statistical features
            try:
                stat_features = self.statistical_extractor.extract_features(code, language)
                features.update(stat_features)
            except Exception as e:
                self.logger.warning(f"Statistical extraction failed for sample {idx}: {e}")
            
            # Token features
            try:
                token_features = self.tokenizer.get_code_metrics(code, language)
                features.update(token_features)
            except Exception as e:
                self.logger.warning(f"Token extraction failed for sample {idx}: {e}")
            
            # Embedding features intentionally disabled to avoid adding random noise
            # The previous implementation added 768 random values which degraded
            # model signal-to-noise. Re-enable only when real embeddings are used.
            
            all_features.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        features_df = features_df.fillna(0)  # Fill NaN values
        
        # Store canonical column order for inference (web app must use same order)
        self.feature_columns = features_df.columns.tolist()
        
        # Prepare features and labels
        self.features = features_df.values
        self.labels = self.training_data['label'].values
        
        self.logger.info(f"Extracted {self.features.shape[1]} features from {len(self.training_data)} samples")
        
        # Save features
        self.data_saver.save_data(features_df, 'data/processed/features.csv', 'csv')
    
    def _train_models(self):
        """Train all models."""
        self.logger.info("Training baseline models")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )
        
        # Train baseline models
        baseline_results = self.baseline_trainer.train_models(X_train, y_train, X_test, y_test)
        self.training_results['baseline'] = baseline_results
        
        # Train transformer model (disabled)
        self.logger.info("Skipping transformer model training")
        # try:
        #     self.transformer_trainer = TransformerTrainer(config.get_section('models'))
        #     
        #     # Convert features back to text for transformer
        #     # This is simplified - in practice, you'd use the original code samples
        #     X_train_text = [f"Code sample {i}" for i in range(len(X_train))]
        #     X_test_text = [f"Code sample {i}" for i in range(len(X_test))]
        #     
        #     transformer_results = self.transformer_trainer.train(
        #         X_train_text, y_train, X_test_text, y_test
        #     )
        #     self.training_results['transformer'] = transformer_results
        #     
        # except Exception as e:
        #     self.logger.error(f"Transformer training failed: {e}")
        #     self.training_results['transformer'] = {'error': str(e)}
    
    def _evaluate_models(self):
        """Evaluate all trained models."""
        self.logger.info("Evaluating models")
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )
        
        # Evaluate baseline models
        for model_name in self.baseline_trainer.models.keys():
            try:
                evaluation = self.baseline_trainer.evaluate_model(X_test, y_test, model_name)
                self.evaluation_results[f'baseline_{model_name}'] = evaluation
                
                # Save to database
                self.db.save_model_performance(
                    f'baseline_{model_name}',
                    {
                        'accuracy': evaluation['accuracy'],
                        'f1_score': evaluation['classification_report']['weighted avg']['f1-score'],
                        'precision': evaluation['classification_report']['weighted avg']['precision'],
                        'recall': evaluation['classification_report']['weighted avg']['recall']
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
    
    def _create_ensemble(self):
        """Create ensemble model."""
        self.logger.info("Creating ensemble model")
        
        # Add baseline models to ensemble
        for model_name, model in self.baseline_trainer.models.items():
            self.ensemble_detector.add_base_model(f'baseline_{model_name}', model)
        
        # Add transformer model if available
        if self.transformer_trainer and self.transformer_trainer.model:
            self.ensemble_detector.add_base_model('transformer', self.transformer_trainer.model)
        
        # Train ensemble
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )
        
        self.ensemble_detector.fit(X_train, y_train, X_test, y_test)
        
        # Evaluate ensemble
        ensemble_predictions = self.ensemble_detector.predict(X_test)
        ensemble_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            y_test, ensemble_predictions
        )
        
        self.evaluation_results['ensemble'] = ensemble_metrics
        
        self.logger.info("Ensemble model created and evaluated")
    
    def _test_robustness(self):
        """Test model robustness against adversarial attacks."""
        self.logger.info("Testing model robustness")
        
        # Use ensemble model for robustness testing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )
        
        # Test ensemble robustness
        robustness_results = self.adversarial_tester.test_ensemble_robustness(
            self.ensemble_detector, X_test, y_test
        )
        
        self.evaluation_results['robustness'] = robustness_results
        
        # Save adversarial test results
        for test_name, results in robustness_results.get('ensemble_results', {}).get('attack_results', {}).items():
            if 'error' not in results:
                self.db.save_adversarial_test(
                    test_name=test_name,
                    attack_type=test_name,
                    original_accuracy=results.get('original_accuracy', 0),
                    adversarial_accuracy=results.get('perturbed_accuracy', 0),
                    model_name='ensemble'
                )
        
        self.logger.info("Robustness testing completed")
    
    def _save_results(self):
        """Save all results and models."""
        self.logger.info("Saving results and models")
        
        # Save baseline models
        self.baseline_trainer.save_models('models/baseline')
        
        # Save canonical feature column list so web app aligns features to training layout
        if getattr(self, 'feature_columns', None):
            models_dir = Path('models')
            models_dir.mkdir(parents=True, exist_ok=True)
            with open(models_dir / 'feature_columns.json', 'w') as f:
                json.dump(self.feature_columns, f, indent=2)
            self.logger.info(f"Saved feature_columns.json ({len(self.feature_columns)} columns)")
        
        # Save transformer model
        if self.transformer_trainer:
            self.transformer_trainer.save_model('models/transformer')
        
        # Save ensemble model
        self.ensemble_detector.save_ensemble('models/ensemble')
        
        # Save evaluation results
        self.data_saver.save_data(
            self.evaluation_results, 
            'results/evaluation_results.json', 
            'json'
        )
        
        # Save training results
        self.data_saver.save_data(
            self.training_results, 
            'results/training_results.json', 
            'json'
        )
        
        # Generate and save performance report
        self._generate_performance_report()
        
        # Export to Power BI
        self._export_to_powerbi()
        
        self.logger.info("All results and models saved successfully")
    
    def _generate_performance_report(self):
        """Generate comprehensive performance report."""
        report = []
        report.append("AI CODE DETECTION SYSTEM - PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model Performance Summary
        report.append("MODEL PERFORMANCE SUMMARY:")
        report.append("-" * 30)
        
        for model_name, results in self.evaluation_results.items():
            if isinstance(results, dict) and 'accuracy' in results:
                report.append(f"{model_name}:")
                report.append(f"  Accuracy: {results['accuracy']:.4f}")
                if 'f1_score' in results:
                    report.append(f"  F1-Score: {results['f1_score']:.4f}")
                if 'precision' in results:
                    report.append(f"  Precision: {results['precision']:.4f}")
                if 'recall' in results:
                    report.append(f"  Recall: {results['recall']:.4f}")
                report.append("")
        
        # Robustness Summary
        if 'robustness' in self.evaluation_results:
            robustness = self.evaluation_results['robustness']
            if 'ensemble_results' in robustness and 'robustness_summary' in robustness['ensemble_results']:
                summary = robustness['ensemble_results']['robustness_summary']
                report.append("ROBUSTNESS SUMMARY:")
                report.append("-" * 20)
                report.append(f"Overall Robustness: {summary.get('overall_robustness', 0):.4f}")
                report.append(f"Average Robustness Score: {summary.get('average_robustness_score', 0):.4f}")
                report.append("")
        
        # Database Statistics
        db_stats = self.db.get_statistics()
        report.append("DATABASE STATISTICS:")
        report.append("-" * 20)
        report.append(f"Total Predictions: {db_stats.get('total_predictions', 0)}")
        report.append(f"Total Code Samples: {db_stats.get('total_code_samples', 0)}")
        report.append(f"Samples by Language: {db_stats.get('samples_by_language', {})}")
        report.append(f"Samples by Label: {db_stats.get('samples_by_label', {})}")
        
        # Save report
        with open('results/performance_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        self.logger.info("Performance report generated")
    
    def _export_to_powerbi(self):
        """Export data to Power BI format."""
        self.logger.info("Exporting data to Power BI format")
        
        try:
            # Export predictions
            predictions_data = self._prepare_predictions_data()
            predictions_file = self.powerbi_exporter.export_predictions_for_powerbi(
                predictions_data, "predictions_data.csv"
            )
            
            # Export model performance
            performance_file = self.powerbi_exporter.export_model_performance_for_powerbi(
                self.evaluation_results, "model_performance.csv"
            )
            
            # Export feature importance
            if self.baseline_trainer.feature_importance:
                features_file = self.powerbi_exporter.export_feature_importance_for_powerbi(
                    self.baseline_trainer.feature_importance, "feature_importance.csv"
                )
            
            # Get training dataset stats
            stats = self.db.get_statistics()
            
            # Export comprehensive dashboard data
            dashboard_files = self.powerbi_exporter.export_comprehensive_dashboard_data(
                predictions_data,
                self.evaluation_results,
                self.baseline_trainer.feature_importance if hasattr(self.baseline_trainer, 'feature_importance') else {},
                stats
            )
            
            # Create Power BI measures
            measures_file = self.powerbi_exporter.create_powerbi_measures_json()
            
            # Generate instructions
            instructions_file = self.powerbi_exporter.generate_powerbi_report_instructions()
            
            self.logger.info(f"Power BI data exported successfully to data/powerbi/")
            self.logger.info(f"Dashboard files: {list(dashboard_files.keys())}")
            
        except Exception as e:
            import traceback
            self.logger.error(f"Power BI export failed: {e}")
            self.logger.error("Full traceback:\n%s", traceback.format_exc())
            print(f"\n⚠️  Power BI export failed (training and models are still saved): {e}")
            print("   Check logs for details. You can ignore this if you do not use Power BI.")
    
    def _prepare_predictions_data(self) -> List[Dict[str, Any]]:
        """Prepare prediction data for Power BI export."""
        predictions = []
        
        # Get predictions from database
        db_stats = self.db.get_statistics()
        total_predictions = db_stats.get('total_predictions', 0)
        
        # Simulate predictions for demonstration
        # In production, retrieve from database
        if self.training_data is not None and len(self.training_data) > 0:
            for idx, row in self.training_data.iterrows():
                prediction = {
                    'id': idx,
                    'code_hash': hashlib.md5(str(row['code']).encode()).hexdigest(),
                    'code_sample': row.get('code', ''),
                    'language': row.get('language', 'unknown'),
                    'prediction': row.get('label', 0),
                    'confidence': np.random.uniform(0.7, 0.95),
                    'model_name': 'ensemble',
                    'timestamp': datetime.now().isoformat()
                }
                predictions.append(prediction)
        
        return predictions

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='AI Code Detection System Training Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--skip-data', action='store_true', help='Skip data collection step')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training step')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluation step')
    parser.add_argument('--skip-adversarial', action='store_true', help='Skip adversarial robustness testing (faster run)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AICodeDetectionPipeline()
    
    try:
        # Run pipeline
        pipeline.run_complete_pipeline(
            skip_data=args.skip_data,
            skip_training=args.skip_training,
            skip_evaluation=args.skip_evaluation,
            skip_adversarial=args.skip_adversarial,
        )
        
        logger.info("Pipeline completed successfully!")
        print("✅ AI Code Detection System training completed successfully!")
        print("📊 Check the 'results/' directory for detailed performance reports")
        print("🤖 Models are saved in the 'models/' directory")
        print("🌐 Run 'streamlit run web_app/app.py' to start the web interface")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
