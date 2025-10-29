"""
Advanced logging utilities for the AI Code Detection System.
Provides structured logging with file rotation and performance tracking.
"""

import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
import json

class PerformanceLogger:
    """Logs performance metrics and timing information."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}
    
    def log_timing(self, operation: str, duration: float, **kwargs):
        """Log timing information for operations."""
        self.logger.info(f"TIMING: {operation} took {duration:.4f}s", extra={
            'operation': operation,
            'duration': duration,
            **kwargs
        })
    
    def log_metric(self, metric_name: str, value: float, **kwargs):
        """Log performance metrics."""
        self.metrics[metric_name] = value
        self.logger.info(f"METRIC: {metric_name}={value:.4f}", extra={
            'metric_name': metric_name,
            'value': value,
            **kwargs
        })
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Log model performance metrics."""
        for metric, value in metrics.items():
            self.log_metric(f"{model_name}_{metric}", value)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all logged metrics."""
        return self.metrics.copy()

def setup_logger(
    name: str = "ai_code_detector",
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Set up structured logger with file rotation."""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_size, backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_function_call(func):
    """Decorator to log function calls with timing."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("ai_code_detector")
        start_time = time.time()
        
        logger.debug(f"Calling {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {duration:.4f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {duration:.4f}s: {str(e)}")
            raise
    
    return wrapper

def log_model_training(model_name: str, metrics: Dict[str, float]):
    """Log model training metrics in structured format."""
    logger = logging.getLogger("ai_code_detector")
    
    log_data = {
        "event": "model_training_complete",
        "model_name": model_name,
        "timestamp": time.time(),
        "metrics": metrics
    }
    
    logger.info(f"Model training completed: {model_name}", extra=log_data)

def log_prediction(prediction: Dict[str, Any], confidence: float, model_name: str):
    """Log prediction results in structured format."""
    logger = logging.getLogger("ai_code_detector")
    
    log_data = {
        "event": "prediction_made",
        "model_name": model_name,
        "prediction": prediction,
        "confidence": confidence,
        "timestamp": time.time()
    }
    
    logger.info(f"Prediction made by {model_name}: {prediction}", extra=log_data)

class StructuredLogger:
    """Logger that outputs structured JSON logs for better analysis."""
    
    def __init__(self, name: str = "ai_code_detector_structured"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log structured event data."""
        self.logger.info(event_type, extra={
            "event_type": event_type,
            "timestamp": time.time(),
            **data
        })

class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs JSON logs."""
    
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

# Global logger instances
main_logger = setup_logger()
performance_logger = PerformanceLogger(main_logger)
structured_logger = StructuredLogger()

def get_logger(name: str = "ai_code_detector") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)

def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    return performance_logger
