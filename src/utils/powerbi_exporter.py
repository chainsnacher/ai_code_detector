"""
Power BI export functionality for visualization and analysis.
Exports data in formats compatible with Power BI.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import hashlib
import numpy as np

logger = logging.getLogger("ai_code_detector")

class PowerBIExporter:
    """Exports data to Power BI compatible formats."""
    
    def __init__(self):
        """Initialize Power BI exporter."""
        self.data_dir = Path("data/powerbi")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def export_predictions_for_powerbi(
        self, 
        predictions: List[Dict[str, Any]],
        output_file: str = "predictions_data.csv"
    ) -> str:
        """
        Export prediction data to CSV for Power BI.
        
        Args:
            predictions: List of prediction dictionaries
            output_file: Output filename
            
        Returns:
            Path to exported file
        """
        if not predictions:
            logger.warning("No predictions to export")
            return ""
        
        # Prepare data for export
        df_data = []
        for pred in predictions:
            df_data.append({
                'prediction_id': pred.get('id', ''),
                'code_hash': pred.get('code_hash', ''),
                'language': pred.get('language', ''),
                'prediction': 'AI Generated' if pred.get('prediction', 0) == 1 else 'Human Written',
                'confidence': pred.get('confidence', 0.0),
                'model_name': pred.get('model_name', ''),
                'timestamp': pred.get('timestamp', ''),
                'code_length': len(pred.get('code_sample', '')),
                'type': self._classify_code_type(pred.get('code_sample', ''))
            })
        
        df = pd.DataFrame(df_data)
        
        # Export to CSV
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(df)} predictions to {output_path}")
        
        return str(output_path)
    
    def export_model_performance_for_powerbi(
        self,
        performance_data: Dict[str, Any],
        output_file: str = "model_performance.csv"
    ) -> str:
        """
        Export model performance metrics to CSV for Power BI.
        
        Args:
            performance_data: Dictionary of model performance metrics
            output_file: Output filename
            
        Returns:
            Path to exported file
        """
        df_data = []
        
        for model_name, metrics in performance_data.items():
            if not isinstance(metrics, dict):
                continue
            # Support both top-level metrics and nested classification_report
            weighted = metrics.get('classification_report', {}).get('weighted avg', {}) if isinstance(metrics.get('classification_report'), dict) else {}
            def _float(v):
                if v is None:
                    return 0.0
                if isinstance(v, (np.floating, np.integer)):
                    return float(v)
                return float(v) if isinstance(v, (int, float)) else 0.0
            df_data.append({
                'model_name': model_name,
                'accuracy': _float(metrics.get('accuracy', 0.0)),
                'f1_score': _float(metrics.get('f1_score') or weighted.get('f1-score', 0.0)),
                'precision': _float(metrics.get('precision') or weighted.get('precision', 0.0)),
                'recall': _float(metrics.get('recall') or weighted.get('recall', 0.0)),
                'roc_auc': _float(metrics.get('roc_auc', 0.0)),
                'timestamp': metrics.get('timestamp', datetime.now().isoformat())
            })
        
        df = pd.DataFrame(df_data)
        
        # Export to CSV
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported model performance to {output_path}")
        
        return str(output_path)
    
    def export_dataset_for_training(
        self,
        samples: List[Dict[str, Any]],
        output_file: str = "training_dataset.csv"
    ) -> str:
        """
        Export labeled dataset for training visualization.
        
        Args:
            samples: List of code samples with labels
            output_file: Output filename
            
        Returns:
            Path to exported file
        """
        df_data = []
        
        for sample in samples:
            df_data.append({
                'sample_id': sample.get('id', ''),
                'code_hash': sample.get('hash', ''),
                'language': sample.get('language', ''),
                'label': 'Human' if sample.get('label', 0) == 0 else 'AI',
                'source': sample.get('source', ''),
                'type': sample.get('type', ''),
                'code_length': len(sample.get('code', '')),
                'created_at': sample.get('created_at', '')
            })
        
        df = pd.DataFrame(df_data)
        
        # Export to CSV
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(df)} training samples to {output_path}")
        
        return str(output_path)
    
    def _feature_importance_to_dict(self, features: Union[Dict[str, float], np.ndarray]) -> Dict[str, float]:
        """Convert feature importance to a dict (handles numpy arrays from sklearn)."""
        if isinstance(features, np.ndarray):
            return {f"feature_{i}": float(x) for i, x in enumerate(features)}
        if isinstance(features, dict):
            return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in features.items()}
        return {}
    
    def export_feature_importance_for_powerbi(
        self,
        feature_importance: Union[Dict[str, Dict[str, float]], Dict[str, np.ndarray]],
        output_file: str = "feature_importance.csv"
    ) -> str:
        """
        Export feature importance data for Power BI.
        
        Args:
            feature_importance: Dictionary of model -> (feature dict or numpy array of importances)
            output_file: Output filename
            
        Returns:
            Path to exported file
        """
        df_data = []
        
        for model_name, features in feature_importance.items():
            features_dict = self._feature_importance_to_dict(features)
            for feature_name, importance in features_dict.items():
                df_data.append({
                    'model_name': model_name,
                    'feature_name': feature_name,
                    'importance': importance,
                    'feature_category': self._categorize_feature(feature_name)
                })
        
        df = pd.DataFrame(df_data)
        
        # Export to CSV
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported feature importance to {output_path}")
        
        return str(output_path)
    
    def export_comprehensive_dashboard_data(
        self,
        predictions: List[Dict[str, Any]],
        model_performance: Dict[str, Any],
        feature_importance: Dict[str, Dict[str, float]],
        training_stats: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Export all data needed for a comprehensive Power BI dashboard.
        
        Args:
            predictions: List of predictions
            model_performance: Model performance metrics
            feature_importance: Feature importance data
            training_stats: Training dataset statistics
            
        Returns:
            Dictionary of exported files and their paths
        """
        exported_files = {}
        
        # Export each data source
        exported_files['predictions'] = self.export_predictions_for_powerbi(
            predictions, "dashboard_predictions.csv"
        )
        
        exported_files['performance'] = self.export_model_performance_for_powerbi(
            model_performance, "dashboard_performance.csv"
        )
        
        exported_files['features'] = self.export_feature_importance_for_powerbi(
            feature_importance, "dashboard_features.csv"
        )
        
        # Export training stats as JSON
        stats_file = self.data_dir / "dashboard_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(training_stats, f, indent=2)
        exported_files['stats'] = str(stats_file)
        
        logger.info(f"Exported comprehensive dashboard data to {self.data_dir}")
        
        return exported_files
    
    def create_powerbi_measures_json(self, output_file: str = "measures.json") -> str:
        """
        Create DAX measures for Power BI dashboard.
        
        Args:
            output_file: Output filename
            
        Returns:
            Path to exported file
        """
        measures = {
            'Total Samples': 'CALCULATE(COUNTROWS(Predictions))',
            'AI Percentage': 'DIVIDE(COUNTROWS(FILTER(Predictions, Predictions[prediction] = "AI Generated")), COUNTROWS(Predictions))',
            'Average Confidence': 'AVERAGE(Predictions[confidence])',
            'High Confidence Samples': 'CALCULATE(COUNTROWS(Predictions), FILTER(Predictions, Predictions[confidence] >= 0.8))',
            'Accuracy by Language': 'AVERAGEX(VALUES(Predictions[language]), AVERAGE(Predictions[confidence]))'
        }
        
        measures_file = self.data_dir / output_file
        with open(measures_file, 'w') as f:
            json.dump(measures, f, indent=2)
        
        logger.info(f"Created Power BI measures file: {measures_file}")
        
        return str(measures_file)
    
    def _classify_code_type(self, code: str) -> str:
        """Classify code by type."""
        if not code:
            return 'unknown'
        
        code_lower = code.lower()
        
        if any(kw in code_lower for kw in ['class ', 'struct ', 'interface ']):
            return 'class_definition'
        elif any(kw in code_lower for kw in ['def ', 'function ', 'fn ', 'public ']):
            return 'function'
        elif 'import' in code_lower or '#include' in code_lower:
            return 'import_statement'
        elif 'main' in code_lower or 'entry' in code_lower:
            return 'main_function'
        elif '#' in code_lower and 'def' in code_lower:
            return 'python_function'
        elif 'public static' in code_lower:
            return 'java_main'
        elif '{' in code_lower and 'function' in code_lower:
            return 'javascript_function'
        else:
            return 'other'
    
    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize feature by type."""
        name_lower = feature_name.lower()
        
        if any(kw in name_lower for kw in ['ast', 'tree', 'node', 'depth']):
            return 'structural'
        elif any(kw in name_lower for kw in ['token', 'keyword', 'lexical']):
            return 'lexical'
        elif any(kw in name_lower for kw in ['comment', 'docstring', 'doc']):
            return 'documentation'
        elif any(kw in name_lower for kw in ['naming', 'variable', 'name']):
            return 'naming'
        elif any(kw in name_lower for kw in ['length', 'size', 'count']):
            return 'statistical'
        elif any(kw in name_lower for kw in ['indent', 'space', 'style']):
            return 'stylistic'
        else:
            return 'other'
    
    def generate_powerbi_report_instructions(self) -> str:
        """Generate instructions for creating Power BI report."""
        instructions = """
# Power BI Dashboard Setup Instructions

## Data Sources
1. Navigate to: data/powerbi/
2. Load the following CSV files into Power BI:
   - dashboard_predictions.csv - Prediction results
   - dashboard_performance.csv - Model performance metrics
   - dashboard_features.csv - Feature importance data
   - dashboard_stats.json - Training statistics

## Recommended Visualizations

### 1. Prediction Distribution
- Chart Type: Donut Chart
- Values: Count by prediction type (AI Generated vs Human Written)
- Colors: Red for AI, Green for Human

### 2. Confidence Distribution
- Chart Type: Histogram
- X-Axis: Confidence levels (0.0 - 1.0)
- Y-Axis: Count of predictions
- Color by: Prediction type

### 3. Model Performance Comparison
- Chart Type: Bar Chart
- X-Axis: Model names
- Y-Axis: Accuracy, F1-Score, Precision, Recall

### 4. Language Distribution
- Chart Type: Pie Chart
- Values: Count by language
- Tooltip: Show percentage

### 5. Feature Importance
- Chart Type: Horizontal Bar Chart
- X-Axis: Importance scores
- Y-Axis: Feature names
- Color by: Feature category

### 6. Time Series Analysis
- Chart Type: Line Chart
- X-Axis: Timestamp
- Y-Axis: Prediction count
- Series: Split by prediction type

## DAX Measures (Load from measures.json)
- Total Samples
- AI Percentage
- Average Confidence
- High Confidence Samples
- Accuracy by Language

## Filters
- Add slicers for:
  - Language
  - Date range
  - Confidence threshold
  - Model name
  - Prediction type

## Dashboard Layout
1. Top Section: Overall metrics (KPIs)
2. Left: Prediction distribution
3. Center: Confidence histogram
4. Right: Model performance comparison
5. Bottom: Language distribution and feature importance
"""
        
        instructions_file = self.data_dir / "POWERBI_INSTRUCTIONS.md"
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        logger.info(f"Created Power BI instructions: {instructions_file}")
        
        return str(instructions_file)

