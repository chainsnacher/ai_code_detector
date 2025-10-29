"""
Database integration for the AI Code Detection System.
Provides SQLite database operations for storing results, models, and metadata.
"""

import sqlite3
import json
import pickle
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from contextlib import contextmanager

logger = logging.getLogger("ai_code_detector")

class DatabaseManager:
    """Manages database operations for the AI Code Detection System."""
    
    def __init__(self, db_path: str = "data/detection_results.db"):
        """Initialize database manager."""
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code_hash TEXT UNIQUE NOT NULL,
                    code_sample TEXT NOT NULL,
                    language TEXT,
                    prediction INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    features TEXT,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    dataset_type TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Feature importance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_importance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    feature_type TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Adversarial test results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adversarial_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_name TEXT NOT NULL,
                    attack_type TEXT NOT NULL,
                    original_accuracy REAL NOT NULL,
                    adversarial_accuracy REAL NOT NULL,
                    robustness_score REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Training sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    dataset_size INTEGER,
                    training_duration REAL,
                    best_accuracy REAL,
                    hyperparameters TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Code samples table (for training data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS code_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code_hash TEXT UNIQUE NOT NULL,
                    code_sample TEXT NOT NULL,
                    language TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    source TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def save_prediction(
        self, 
        code_hash: str, 
        code_sample: str, 
        prediction: int, 
        confidence: float, 
        model_name: str,
        language: Optional[str] = None,
        features: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Save prediction result to database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO predictions 
                (code_hash, code_sample, language, prediction, confidence, model_name, features, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                code_hash,
                code_sample,
                language,
                prediction,
                confidence,
                model_name,
                json.dumps(features) if features else None,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_prediction(self, code_hash: str) -> Optional[Dict]:
        """Get prediction by code hash."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM predictions WHERE code_hash = ?
            """, (code_hash,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def save_model_performance(
        self, 
        model_name: str, 
        metrics: Dict[str, float], 
        dataset_type: str = "test"
    ):
        """Save model performance metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for metric_name, metric_value in metrics.items():
                cursor.execute("""
                    INSERT INTO model_performance 
                    (model_name, metric_name, metric_value, dataset_type)
                    VALUES (?, ?, ?, ?)
                """, (model_name, metric_name, metric_value, dataset_type))
            
            conn.commit()
    
    def get_model_performance(self, model_name: str) -> pd.DataFrame:
        """Get model performance metrics as DataFrame."""
        with self._get_connection() as conn:
            query = """
                SELECT metric_name, metric_value, dataset_type, timestamp
                FROM model_performance 
                WHERE model_name = ?
                ORDER BY timestamp DESC
            """
            return pd.read_sql_query(query, conn, params=(model_name,))
    
    def save_feature_importance(
        self, 
        model_name: str, 
        feature_importance: Dict[str, float],
        feature_type: str = "statistical"
    ):
        """Save feature importance scores."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for feature_name, importance_score in feature_importance.items():
                cursor.execute("""
                    INSERT INTO feature_importance 
                    (model_name, feature_name, importance_score, feature_type)
                    VALUES (?, ?, ?, ?)
                """, (model_name, feature_name, importance_score, feature_type))
            
            conn.commit()
    
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        with self._get_connection() as conn:
            query = """
                SELECT feature_name, importance_score, feature_type, timestamp
                FROM feature_importance 
                WHERE model_name = ?
                ORDER BY importance_score DESC
            """
            return pd.read_sql_query(query, conn, params=(model_name,))
    
    def save_adversarial_test(
        self, 
        test_name: str, 
        attack_type: str, 
        original_accuracy: float, 
        adversarial_accuracy: float, 
        model_name: str
    ):
        """Save adversarial test results."""
        robustness_score = adversarial_accuracy / original_accuracy if original_accuracy > 0 else 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO adversarial_tests 
                (test_name, attack_type, original_accuracy, adversarial_accuracy, robustness_score, model_name)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (test_name, attack_type, original_accuracy, adversarial_accuracy, robustness_score, model_name))
            
            conn.commit()
    
    def save_training_session(
        self, 
        session_name: str, 
        model_type: str, 
        dataset_size: int, 
        training_duration: float, 
        best_accuracy: float, 
        hyperparameters: Dict[str, Any]
    ):
        """Save training session information."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO training_sessions 
                (session_name, model_type, dataset_size, training_duration, best_accuracy, hyperparameters)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_name, model_type, dataset_size, training_duration, best_accuracy, json.dumps(hyperparameters)))
            
            conn.commit()
    
    def save_code_sample(
        self, 
        code_hash: str, 
        code_sample: str, 
        language: str, 
        label: int, 
        source: str = "unknown",
        metadata: Optional[Dict] = None
    ):
        """Save code sample for training data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO code_samples 
                (code_hash, code_sample, language, label, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (code_hash, code_sample, language, label, source, json.dumps(metadata) if metadata else None))
            
            conn.commit()
    
    def get_training_data(self, language: Optional[str] = None) -> pd.DataFrame:
        """Get training data as DataFrame."""
        with self._get_connection() as conn:
            if language:
                query = """
                    SELECT * FROM code_samples 
                    WHERE language = ?
                    ORDER BY created_at DESC
                """
                return pd.read_sql_query(query, conn, params=(language,))
            else:
                query = """
                    SELECT * FROM code_samples 
                    ORDER BY created_at DESC
                """
                return pd.read_sql_query(query, conn)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            stats["total_predictions"] = cursor.fetchone()[0]
            
            # Count code samples
            cursor.execute("SELECT COUNT(*) FROM code_samples")
            stats["total_code_samples"] = cursor.fetchone()[0]
            
            # Count by language
            cursor.execute("""
                SELECT language, COUNT(*) as count 
                FROM code_samples 
                GROUP BY language
            """)
            stats["samples_by_language"] = dict(cursor.fetchall())
            
            # Count by label
            cursor.execute("""
                SELECT label, COUNT(*) as count 
                FROM code_samples 
                GROUP BY label
            """)
            stats["samples_by_label"] = dict(cursor.fetchall())
            
            # Model performance summary
            cursor.execute("""
                SELECT model_name, AVG(metric_value) as avg_accuracy
                FROM model_performance 
                WHERE metric_name = 'accuracy'
                GROUP BY model_name
            """)
            stats["model_performance"] = dict(cursor.fetchall())
            
            return stats
    
    def export_data(self, output_path: str, table_name: str = "predictions"):
        """Export table data to CSV."""
        with self._get_connection() as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} rows from {table_name} to {output_path}")
    
    def clear_old_data(self, days: int = 30):
        """Clear old data older than specified days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear old predictions
            cursor.execute("""
                DELETE FROM predictions 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))
            
            deleted_predictions = cursor.rowcount
            
            # Clear old model performance records
            cursor.execute("""
                DELETE FROM model_performance 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days))
            
            deleted_performance = cursor.rowcount
            
            conn.commit()
            
            logger.info(f"Cleared {deleted_predictions} old predictions and {deleted_performance} old performance records")
            
            return deleted_predictions + deleted_performance

# Global database manager instance
db_manager = DatabaseManager()

def get_database() -> DatabaseManager:
    """Get global database manager instance."""
    return db_manager
