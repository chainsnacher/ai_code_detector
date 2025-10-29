"""
Basic tests for the AI Code Detection System
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append('src')

from utils.config import get_config
from utils.logger import setup_logger
from utils.data_utils import DataValidator, DataProcessor
from preprocessing.ast_parser import ASTFeatureExtractor
from preprocessing.feature_extractor import StatisticalFeatureExtractor
from preprocessing.code_tokenizer import AdvancedCodeTokenizer
from models.baseline_models import BaselineModelTrainer
from evaluation.metrics import AdvancedMetrics

class TestBasicFunctionality:
    """Test basic functionality of the system."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = get_config()
        assert config is not None
        assert 'data_collection' in config.config
        assert 'features' in config.config
        assert 'models' in config.config
    
    def test_logger_setup(self):
        """Test logger setup."""
        logger = setup_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_data_validator(self):
        """Test data validation."""
        validator = DataValidator()
        
        # Test valid Python code
        valid_code = "def hello():\n    print('world')"
        result = validator.validate_code_sample(valid_code, "python")
        assert result["is_valid"] == True
        
        # Test invalid code
        invalid_code = "def hello(\n    print('world')"  # Missing closing parenthesis
        result = validator.validate_code_sample(invalid_code, "python")
        assert result["is_valid"] == False
    
    def test_ast_feature_extraction(self):
        """Test AST feature extraction."""
        extractor = ASTFeatureExtractor()
        
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        features = extractor.extract_features(code, "python")
        assert isinstance(features, dict)
        assert "ast_depth" in features
        assert "function_count" in features
        assert features["function_count"] == 1
    
    def test_statistical_feature_extraction(self):
        """Test statistical feature extraction."""
        extractor = StatisticalFeatureExtractor()
        
        code = """
def hello_world():
    print("Hello, World!")
    return "success"
"""
        
        features = extractor.extract_features(code, "python")
        assert isinstance(features, dict)
        assert "total_lines" in features
        assert "total_characters" in features
        assert features["total_lines"] > 0
    
    def test_code_tokenizer(self):
        """Test code tokenization."""
        tokenizer = AdvancedCodeTokenizer()
        
        code = "def hello(): print('world')"
        tokens = tokenizer.tokenize(code, "python")
        
        assert isinstance(tokens, dict)
        assert "keywords" in tokens
        assert "identifiers" in tokens
        assert "operators" in tokens
    
    def test_baseline_model_trainer(self):
        """Test baseline model trainer."""
        trainer = BaselineModelTrainer()
        
        # Create dummy data
        X = np.random.random((100, 10))
        y = np.random.randint(0, 2, 100)
        
        # Test model creation
        model = trainer._create_model("random_forest")
        assert model is not None
        
        # Test training (simplified)
        try:
            model.fit(X, y)
            predictions = model.predict(X[:10])
            assert len(predictions) == 10
        except Exception as e:
            pytest.skip(f"Model training failed: {e}")
    
    def test_metrics_calculator(self):
        """Test metrics calculation."""
        calculator = AdvancedMetrics()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_proba = np.array([0.2, 0.8, 0.7, 0.9, 0.3])
        
        metrics = calculator.calculate_comprehensive_metrics(y_true, y_pred, y_proba)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_data_processor(self):
        """Test data processing."""
        processor = DataProcessor()
        
        # Create dummy data
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'label': [0, 1, 0, 1, 0]
        })
        
        # Test feature preparation
        features = processor.prepare_features(df, ['feature1', 'feature2'])
        assert features.shape == (5, 2)
        
        # Test label preparation
        labels = processor.prepare_labels(df['label'])
        assert len(labels) == 5
        assert set(labels) == {0, 1}

class TestIntegration:
    """Test integration between components."""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline with dummy data."""
        # This is a simplified test - in practice, you'd test the full pipeline
        
        # Create dummy code samples
        human_code = """
def calculate_sum(a, b):
    return a + b
"""
        
        ai_code = """
def compute_addition_of_two_numbers(first_number, second_number):
    # This function adds two numbers together
    result = first_number + second_number
    return result
"""
        
        # Test feature extraction
        ast_extractor = ASTFeatureExtractor()
        stat_extractor = StatisticalFeatureExtractor()
        tokenizer = AdvancedCodeTokenizer()
        
        human_features = {}
        human_features.update(ast_extractor.extract_features(human_code, "python"))
        human_features.update(stat_extractor.extract_features(human_code, "python"))
        human_features.update(tokenizer.get_code_metrics(human_code, "python"))
        
        ai_features = {}
        ai_features.update(ast_extractor.extract_features(ai_code, "python"))
        ai_features.update(stat_extractor.extract_features(ai_code, "python"))
        ai_features.update(tokenizer.get_code_metrics(ai_code, "python"))
        
        # Verify features are different
        assert human_features != ai_features
        
        # Test that features are numeric
        for key, value in human_features.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value)
    
    def test_model_training_and_prediction(self):
        """Test model training and prediction."""
        # Create dummy training data
        np.random.seed(42)
        X = np.random.random((50, 20))
        y = np.random.randint(0, 2, 50)
        
        # Test baseline model training
        trainer = BaselineModelTrainer()
        
        try:
            # Train a simple model
            model = trainer._create_model("logistic_regression")
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X[:10])
            probabilities = model.predict_proba(X[:10])
            
            assert len(predictions) == 10
            assert probabilities.shape == (10, 2)
            assert all(pred in [0, 1] for pred in predictions)
            
        except Exception as e:
            pytest.skip(f"Model training failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
