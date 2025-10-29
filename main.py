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
from datetime import datetime

# Add src to path
sys.path.append('src')

from utils.config import get_config
from utils.logger import setup_logger, get_performance_logger
from utils.database import get_database
from utils.data_utils import DataProcessor, DataSaver, DataBalancer
# from utils.data_labeler import DataLabeler  # Temporarily disabled
from utils.powerbi_exporter import PowerBIExporter
from preprocessing.language_detector import LanguageDetector

# Feature extraction
from preprocessing.ast_parser import ASTFeatureExtractor
from preprocessing.feature_extractor import StatisticalFeatureExtractor
from preprocessing.embedding_generator import CodeEmbeddingGenerator, EmbeddingEnsemble
from preprocessing.code_tokenizer import AdvancedCodeTokenizer

# Models
from models.baseline_models import BaselineModelTrainer, AdvancedEnsemble
from models.transformer_model import TransformerTrainer, MultiModelEnsemble
from models.ensemble_model import AdvancedEnsembleDetector

# Evaluation
from evaluation.metrics import AdvancedMetrics
from evaluation.cross_validation import AdvancedCrossValidator
from evaluation.adversarial_testing import AdversarialTester

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
        self.embedding_generator = CodeEmbeddingGenerator(config.get_section('features'))
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
    
    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline."""
        self.logger.info("Starting AI Code Detection Pipeline")
        start_time = time.time()
        
        try:
            # Step 1: Data Collection and Preparation
            self.logger.info("Step 1: Data Collection and Preparation")
            self._collect_and_prepare_data()
            
            # Step 2: Feature Engineering
            self.logger.info("Step 2: Feature Engineering")
            self._extract_features()
            
            # Step 3: Model Training
            self.logger.info("Step 3: Model Training")
            self._train_models()
            
            # Step 4: Model Evaluation
            self.logger.info("Step 4: Model Evaluation")
            self._evaluate_models()
            
            # Step 5: Ensemble Creation
            self.logger.info("Step 5: Ensemble Creation")
            self._create_ensemble()
            
            # Step 6: Adversarial Testing
            self.logger.info("Step 6: Adversarial Testing")
            self._test_robustness()
            
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
        
        # Generate synthetic code samples
        human_code_samples = self._generate_human_code_samples(1000)
        ai_code_samples = self._generate_ai_code_samples(1000)
        
        # Combine and label data
        all_samples = human_code_samples + ai_code_samples
        labels = [0] * len(human_code_samples) + [1] * len(ai_code_samples)
        
        # Create DataFrame
        self.training_data = pd.DataFrame({
            'code': all_samples,
            'label': labels,
            'language': ['python'] * len(all_samples)
        })
        
        # Balance dataset if needed
        if config.get('data_balancing', {}).get('enabled', False):
            self.training_data = self.data_balancer.balance_dataset(
                self.training_data, 'label', method='undersample'
            )
        
        self.logger.info(f"Created dataset with {len(self.training_data)} samples")
        
        # Save to database
        for idx, row in self.training_data.iterrows():
            self.db.save_code_sample(
                code_hash=f"sample_{idx}",
                code_sample=row['code'],
                language=row['language'],
                label=row['label'],
                source='synthetic'
            )
    
    def _generate_human_code_samples(self, n_samples: int) -> List[str]:
        """Generate synthetic human-like code samples."""
        samples = []
        
        # Simple function templates
        function_templates = [
            """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
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
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
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
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
""",
            """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""
        ]
        
        for i in range(n_samples):
            template = function_templates[i % len(function_templates)]
            # Add some variation
            if i % 3 == 0:
                template = template.replace('def ', 'def my_')
            elif i % 3 == 1:
                template = template.replace('    ', '  ')
            
            samples.append(template.strip())
        
        return samples
    
    def _generate_ai_code_samples(self, n_samples: int) -> List[str]:
        """Generate synthetic AI-like code samples."""
        samples = []
        
        # AI-like code patterns (more verbose, different style)
        ai_templates = [
            """
def calculate_fibonacci_sequence(n):
    # This function calculates the nth Fibonacci number
    if n <= 1:
        return n
    else:
        return calculate_fibonacci_sequence(n-1) + calculate_fibonacci_sequence(n-2)
""",
            """
def perform_bubble_sort_algorithm(input_array):
    array_length = len(input_array)
    for i in range(array_length):
        for j in range(0, array_length - i - 1):
            if input_array[j] > input_array[j + 1]:
                input_array[j], input_array[j + 1] = input_array[j + 1], input_array[j]
    return input_array
""",
            """
def execute_binary_search_algorithm(sorted_array, target_value):
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
    def __init__(self, data_value):
        self.data_value = data_value
        self.next_node = None

class LinkedListDataStructure:
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
    if number < 2:
        return False
    for divisor in range(2, int(number**0.5) + 1):
        if number % divisor == 0:
            return False
    return True
"""
        ]
        
        for i in range(n_samples):
            template = ai_templates[i % len(ai_templates)]
            # Add some variation
            if i % 4 == 0:
                template = template.replace('def ', 'def compute_')
            elif i % 4 == 1:
                template = template.replace('    ', '        ')
            
            samples.append(template.strip())
        
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
        
        # Train transformer model
        self.logger.info("Training transformer model")
        try:
            self.transformer_trainer = TransformerTrainer(config.get_section('models'))
            
            # Convert features back to text for transformer
            # This is simplified - in practice, you'd use the original code samples
            X_train_text = [f"Code sample {i}" for i in range(len(X_train))]
            X_test_text = [f"Code sample {i}" for i in range(len(X_test))]
            
            transformer_results = self.transformer_trainer.train(
                X_train_text, y_train, X_test_text, y_test
            )
            self.training_results['transformer'] = transformer_results
            
        except Exception as e:
            self.logger.error(f"Transformer training failed: {e}")
            self.training_results['transformer'] = {'error': str(e)}
    
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
            self.logger.error(f"Error exporting to Power BI: {e}")
    
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
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AICodeDetectionPipeline()
    
    try:
        # Run pipeline
        pipeline.run_complete_pipeline()
        
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
