"""
Data utilities for the AI Code Detection System.
Provides helper functions for data processing, validation, and manipulation.
"""

import pandas as pd
import numpy as np
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger("ai_code_detector")

class DataValidator:
    """Validates data quality and consistency."""
    
    @staticmethod
    def validate_code_sample(code: str, language: str = "python") -> Dict[str, Any]:
        """Validate a single code sample."""
        validation_result = {
            "is_valid": True,
            "issues": [],
            "metrics": {}
        }
        
        # Basic validation
        if not code or len(code.strip()) == 0:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Empty code sample")
            return validation_result
        
        # Length validation
        code_length = len(code)
        validation_result["metrics"]["length"] = code_length
        
        if code_length < 10:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Code too short (< 10 characters)")
        elif code_length > 50000:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Code too long (> 50,000 characters)")
        
        # Language-specific validation
        if language == "python":
            py_res = DataValidator._validate_python_code(code)
            # Merge issues
            validation_result["issues"].extend(py_res.get("issues", []))
            # Merge metrics without overwriting existing ones (like length)
            validation_result["metrics"].update(py_res.get("metrics", {}))
            # Combine validity flags
            if "is_valid" in py_res:
                validation_result["is_valid"] = validation_result["is_valid"] and py_res["is_valid"]
        elif language == "java":
            java_res = DataValidator._validate_java_code(code)
            validation_result["issues"].extend(java_res.get("issues", []))
            validation_result["metrics"].update(java_res.get("metrics", {}))
        elif language == "javascript":
            js_res = DataValidator._validate_javascript_code(code)
            validation_result["issues"].extend(js_res.get("issues", []))
            validation_result["metrics"].update(js_res.get("metrics", {}))
        
        return validation_result
    
    @staticmethod
    def _validate_python_code(code: str) -> Dict[str, Any]:
        """Validate Python-specific code patterns."""
        result = {"issues": [], "metrics": {}, "is_valid": True}
        
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            result["issues"].append(f"Python syntax error: {str(e)}")
            # Mark invalid when a syntax error is present
            result["is_valid"] = False
        
        # Check for basic Python patterns
        if "def " in code:
            result["metrics"]["has_functions"] = True
        if "class " in code:
            result["metrics"]["has_classes"] = True
        if "import " in code:
            result["metrics"]["has_imports"] = True

        return result
    
    @staticmethod
    def _validate_java_code(code: str) -> Dict[str, Any]:
        """Validate Java-specific code patterns."""
        result = {"issues": [], "metrics": {}}
        
        # Check for basic Java patterns
        if "public class" in code or "class " in code:
            result["metrics"]["has_classes"] = True
        if "public static void main" in code:
            result["metrics"]["has_main_method"] = True
        if "import " in code:
            result["metrics"]["has_imports"] = True
        
        return result
    
    @staticmethod
    def _validate_javascript_code(code: str) -> Dict[str, Any]:
        """Validate JavaScript-specific code patterns."""
        result = {"issues": [], "metrics": {}}
        
        # Check for basic JavaScript patterns
        if "function " in code or "=>" in code:
            result["metrics"]["has_functions"] = True
        if "class " in code:
            result["metrics"]["has_classes"] = True
        if "require(" in code or "import " in code:
            result["metrics"]["has_imports"] = True
        
        return result

class DataProcessor:
    """Processes and transforms data for model training."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def prepare_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Prepare features for model training."""
        if not self.is_fitted:
            # Fit scaler on training data
            self.scaler.fit(df[feature_columns])
            self.is_fitted = True
        
        # Transform features
        features = self.scaler.transform(df[feature_columns])
        return features
    
    def prepare_labels(self, labels: List[str]) -> np.ndarray:
        """Prepare labels for model training."""
        return self.label_encoder.fit_transform(labels)
    
    def inverse_transform_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """Convert encoded labels back to original labels."""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def create_train_test_split(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create stratified train-test split."""
        return train_test_split(
            features, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )

class CodePreprocessor:
    """Preprocesses code samples for feature extraction."""
    
    @staticmethod
    def clean_code(code: str) -> str:
        """Clean and normalize code sample."""
        # Remove excessive whitespace
        code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
        
        # Normalize line endings
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove trailing whitespace
        lines = [line.rstrip() for line in code.split('\n')]
        code = '\n'.join(lines)
        
        return code.strip()
    
    @staticmethod
    def extract_metadata(code: str) -> Dict[str, Any]:
        """Extract metadata from code sample."""
        lines = code.split('\n')
        
        metadata = {
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "total_characters": len(code),
            "total_words": len(code.split()),
            "indentation_style": CodePreprocessor._detect_indentation_style(code),
            "line_length_stats": CodePreprocessor._calculate_line_length_stats(lines)
        }
        
        return metadata
    
    @staticmethod
    def _detect_indentation_style(code: str) -> str:
        """Detect whether code uses tabs or spaces for indentation."""
        lines = [line for line in code.split('\n') if line.strip() and line.startswith((' ', '\t'))]
        
        if not lines:
            return "unknown"
        
        tab_count = sum(1 for line in lines if line.startswith('\t'))
        space_count = sum(1 for line in lines if line.startswith(' '))
        
        if tab_count > space_count:
            return "tabs"
        elif space_count > tab_count:
            return "spaces"
        else:
            return "mixed"
    
    @staticmethod
    def _calculate_line_length_stats(lines: List[str]) -> Dict[str, float]:
        """Calculate statistics about line lengths."""
        line_lengths = [len(line) for line in lines if line.strip()]
        
        if not line_lengths:
            return {"mean": 0, "std": 0, "max": 0, "min": 0}
        
        return {
            "mean": np.mean(line_lengths),
            "std": np.std(line_lengths),
            "max": np.max(line_lengths),
            "min": np.min(line_lengths)
        }

class DataSaver:
    """Saves and loads processed data."""
    
    @staticmethod
    def save_data(data: Any, filepath: str, format: str = "pickle"):
        """Save data to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif format == "json":
            # Convert numpy/pandas objects to JSON-serializable Python types
            def _convert(obj):
                # numpy arrays
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                # numpy scalar types
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                # pandas DataFrame/Series
                if isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient='records')
                if isinstance(obj, pd.Series):
                    return obj.tolist()
                # dicts, lists, tuples
                if isinstance(obj, dict):
                    return {k: _convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_convert(i) for i in obj]
                if isinstance(obj, tuple):
                    return [_convert(i) for i in obj]
                if isinstance(obj, set):
                    return [_convert(i) for i in obj]
                # scikit-learn estimators (and other objects exposing get_params)
                if hasattr(obj, 'get_params') and callable(getattr(obj, 'get_params')):
                    try:
                        params = obj.get_params()
                    except Exception:
                        params = {}
                    return {
                        '__sklearn_estimator__': obj.__class__.__name__,
                        'params': _convert(params)
                    }
                # callables (functions, lambdas) - represent by name
                if callable(obj):
                    try:
                        return getattr(obj, '__name__', str(obj))
                    except Exception:
                        return str(obj)
                # Path objects
                if isinstance(obj, Path):
                    return str(obj)
                # Fallback
                return obj

            serializable = _convert(data)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
        elif format == "csv" and isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data saved to {filepath}")
    
    @staticmethod
    def load_data(filepath: str, format: str = "pickle") -> Any:
        """Load data from file."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format == "pickle":
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif format == "csv":
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

class DataBalancer:
    """Balances datasets to handle class imbalance."""
    
    @staticmethod
    def balance_dataset(df: pd.DataFrame, target_column: str, method: str = "undersample") -> pd.DataFrame:
        """Balance dataset using specified method."""
        if method == "undersample":
            return DataBalancer._undersample(df, target_column)
        elif method == "oversample":
            return DataBalancer._oversample(df, target_column)
        else:
            raise ValueError(f"Unknown balancing method: {method}")
    
    @staticmethod
    def _undersample(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Undersample majority class."""
        class_counts = df[target_column].value_counts()
        min_count = class_counts.min()
        
        balanced_dfs = []
        for class_value in class_counts.index:
            class_df = df[df[target_column] == class_value]
            if len(class_df) > min_count:
                class_df = class_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True)
    
    @staticmethod
    def _oversample(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Oversample minority class."""
        class_counts = df[target_column].value_counts()
        max_count = class_counts.max()
        
        balanced_dfs = []
        for class_value in class_counts.index:
            class_df = df[df[target_column] == class_value]
            if len(class_df) < max_count:
                # Repeat samples to reach max_count
                repeat_times = max_count // len(class_df)
                remainder = max_count % len(class_df)
                
                repeated_df = pd.concat([class_df] * repeat_times, ignore_index=True)
                if remainder > 0:
                    additional_df = class_df.sample(n=remainder, random_state=42)
                    repeated_df = pd.concat([repeated_df, additional_df], ignore_index=True)
                
                balanced_dfs.append(repeated_df)
            else:
                balanced_dfs.append(class_df)
        
        return pd.concat(balanced_dfs, ignore_index=True)

def generate_code_hash(code: str) -> str:
    """Generate unique hash for code sample."""
    return hashlib.md5(code.encode('utf-8')).hexdigest()

def detect_language(filename: str) -> str:
    """Detect programming language from filename."""
    extension_map = {
        '.py': 'python',
        '.java': 'java',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala'
    }
    
    suffix = Path(filename).suffix.lower()
    return extension_map.get(suffix, 'unknown')
