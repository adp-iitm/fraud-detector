# ml_models/config/config.py

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "saved" / "models"
VECTORIZERS_DIR = PROJECT_ROOT / "saved" / "vectorizers"
SCALERS_DIR = PROJECT_ROOT / "saved" / "scalers"

# Model configurations
MODEL_CONFIG = {
    "url": {
        "model_type": "xgboost",
        "model_file": "url_classifier.pkl",
        "scaler_file": "url_scaler.pkl",
        "features": [
            "url_length", "num_digits", "num_special_chars", "num_dots",
            "num_hyphens", "num_underscores", "num_slashes", "num_percent",
            "has_ip", "is_https", "domain_length", "tld_length",
            "num_subdomains", "suspicious_words_count"
        ],
        "hyperparams": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "random_state": 42
        }
    },
    "email": {
        "model_type": "logistic_regression",
        "model_file": "email_classifier.pkl", 
        "vectorizer_file": "email_vectorizer.pkl",
        "max_features": 10000,
        "hyperparams": {
            "C": 1.0,
            "random_state": 42,
            "max_iter": 1000
        }
    },
    "transaction": {
        "model_type": "lightgbm",
        "model_file": "transaction_classifier.pkl",
        "scaler_file": "transaction_scaler.pkl",
        "features": [
            "amount", "hour", "day_of_week", "is_weekend",
            "merchant_category", "user_age_days", "time_since_last_transaction",
            "transaction_frequency_1h", "transaction_frequency_24h",
            "avg_amount_7d", "std_amount_7d", "amount_zscore"
        ],
        "hyperparams": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "random_state": 42
        }
    }
}

# Data collection URLs
DATA_SOURCES = {
    "phishtank_url": "http://data.phishtank.com/data/online-valid.csv",
    "openphish_url": "https://openphish.com/feed.txt",
    "kaggle_datasets": {
        "phishing_urls": "akashkr/phishing-website-dataset",
        "spam_emails": "uciml/sms-spam-collection-dataset", 
        "fraud_transactions": "mlg-ulb/creditcardfraud"
    }
}

# Inference thresholds
PREDICTION_THRESHOLDS = {
    "url": 0.5,
    "email": 0.5,
    "transaction": 0.3  # Lower threshold for fraud detection
}

# Performance requirements
PERFORMANCE_CONFIG = {
    "max_inference_time_ms": 200,
    "batch_size": 32,
    "enable_onnx_optimization": True
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    "rotation": "1 week",
    "retention": "1 month"
}