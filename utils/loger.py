# ml_models/utils/logger.py

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class MLLogger:
    """Custom logger for ML operations"""
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def critical(self, message: str):
        self.logger.critical(message)

# ml_models/utils/helpers.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Union, Any
import json
import time
from functools import wraps

def timeit(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"{func.__name__} executed in {execution_time:.2f}ms")
        return result
    return wrapper

def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save dictionary to JSON file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default value when denominator is zero"""
    return numerator / denominator if denominator != 0 else default

def normalize_text(text: str) -> str:
    """Normalize text for consistent processing"""
    if not text:
        return ""
    
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    
    # Replace multiple spaces with single space
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text

def batch_process(items: List[Any], batch_size: int = 32):
    """Generator to process items in batches"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def get_file_size(file_path: Union[str, Path]) -> str:
    """Get human-readable file size"""
    path = Path(file_path)
    if not path.exists():
        return "File not found"
    
    size_bytes = path.stat().st_size
    
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def memory_usage():
    """Get current memory usage"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': f"{memory_info.rss / 1024 / 1024:.2f} MB",  # Physical memory
        'vms': f"{memory_info.vms / 1024 / 1024:.2f} MB",  # Virtual memory
        'percent': f"{process.memory_percent():.2f}%"
    }

class ModelRegistry:
    """Simple model registry for managing trained models"""
    
    def __init__(self, registry_path: Union[str, Path]):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry or create new one
        if self.registry_path.exists():
            self.registry = load_json(self.registry_path)
        else:
            self.registry = {}
    
    def register_model(self, model_name: str, model_path: str, 
                      metrics: Dict[str, float], metadata: Dict[str, Any] = None):
        """Register a trained model"""
        model_info = {
            'model_path': str(model_path),
            'metrics': metrics,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat(),
            'file_size': get_file_size(model_path)
        }
        
        self.registry[model_name] = model_info
        self.save_registry()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a registered model"""
        return self.registry.get(model_name, {})
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self.registry.keys())
    
    def get_best_model(self, metric: str = 'roc_auc') -> str:
        """Get the best model based on specified metric"""
        if not self.registry:
            return None
        
        best_model = None
        best_score = -1
        
        for model_name, info in self.registry.items():
            score = info.get('metrics', {}).get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def save_registry(self):
        """Save registry to file"""
        save_json(self.registry, self.registry_path)
    
    def remove_model(self, model_name: str):
        """Remove model from registry"""
        if model_name in self.registry:
            del self.registry[model_name]
            self.save_registry()

class ConfigManager:
    """Configuration management utility"""
    
    def __init__(self, config_file: Union[str, Path]):
        self.config_file = Path(config_file)
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            self.config = load_json(self.config_file)
        else:
            print(f"Configuration file {self.config_file} not found, using defaults")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        save_json(self.config, self.config_file)

def validate_input_data(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """Validate input data has required fields"""
    errors = []
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif data[field] is None or data[field] == "":
            errors.append(f"Field '{field}' cannot be empty")
    
    return errors

def calculate_feature_importance_summary(feature_names: List[str], 
                                       importances: np.ndarray, 
                                       top_n: int = 10) -> Dict[str, float]:
    """Calculate and return top N feature importances"""
    if len(feature_names) != len(importances):
        raise ValueError("Feature names and importances must have same length")
    
    # Sort by importance (descending)
    sorted_indices = np.argsort(importances)[::-1]
    
    # Get top N features
    top_indices = sorted_indices[:top_n]
    
    return {
        feature_names[i]: float(importances[i]) 
        for i in top_indices
    }

def create_model_metadata(model_name: str, model_type: str, 
                         training_data_size: int, features_count: int) -> Dict[str, Any]:
    """Create standard model metadata"""
    return {
        'model_name': model_name,
        'model_type': model_type,
        'training_data_size': training_data_size,
        'features_count': features_count,
        'created_at': datetime.now().isoformat(),
        'python_version': sys.version,
        'framework_versions': {
            'scikit-learn': get_package_version('scikit-learn'),
            'pandas': get_package_version('pandas'),
            'numpy': get_package_version('numpy')
        }
    }

def get_package_version(package_name: str) -> str:
    """Get version of installed package"""
    try:
        import pkg_resources
        return pkg_resources.get_distribution(package_name).version
    except:
        return "unknown"

class PerformanceMonitor:
    """Monitor model performance metrics"""
    
    def __init__(self):
        self.predictions = []
        self.response_times = []
    
    def record_prediction(self, prediction_time: float, model_type: str, 
                         prediction: str, confidence: float):
        """Record a prediction for monitoring"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'prediction_time_ms': prediction_time,
            'prediction': prediction,
            'confidence': confidence
        }
        
        self.predictions.append(record)
        self.response_times.append(prediction_time)
        
        # Keep only last 1000 predictions to manage memory
        if len(self.predictions) > 1000:
            self.predictions = self.predictions[-1000:]
            self.response_times = self.response_times[-1000:]
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.response_times:
            return {}
        
        return {
            'total_predictions': len(self.predictions),
            'avg_response_time_ms': np.mean(self.response_times),
            'median_response_time_ms': np.median(self.response_times),
            'p95_response_time_ms': np.percentile(self.response_times, 95),
            'p99_response_time_ms': np.percentile(self.response_times, 99),
            'max_response_time_ms': np.max(self.response_times),
            'min_response_time_ms': np.min(self.response_times)
        }

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test logger
    logger = MLLogger("test_logger", "logs/test.log")
    logger.info("Test log message")
    
    # Test model registry
    registry = ModelRegistry("models/registry.json")
    print(f"Models in registry: {registry.list_models()}")
    
    # Test memory usage
    print(f"Memory usage: {memory_usage()}")
    
    # Test performance monitor
    performance_monitor.record_prediction(150.5, "url", "phishing", 0.85)
    print(f"Performance stats: {performance_monitor.get_performance_stats()}")
    
    print("Utilities testing complete!")