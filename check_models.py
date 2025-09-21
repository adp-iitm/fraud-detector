#!/usr/bin/env python3
import json
from pathlib import Path
import joblib

def check_models():
    print("Checking ML Models Status")
    print("=" * 50)
    
    model_path = Path("ml_models/saved/models")
    reports_path = Path("ml_models/evaluation/reports")
    
    # Check if models exist
    models = {
        'url_classifier.pkl': 'URL Phishing Detection',
        'email_classifier.pkl': 'Email Phishing Detection', 
        'transaction_classifier.pkl': 'Transaction Fraud Detection'
    }
    
    print("\nModel Files:")
    for model_file, description in models.items():
        model_filepath = model_path / model_file
        if model_filepath.exists():
            size_mb = model_filepath.stat().st_size / (1024 * 1024)
            print(f"SUCCESS: {description}: {size_mb:.2f} MB")
        else:
            print(f"MISSING: {description}: Not found")
    
    # Check training summary
    summary_file = reports_path / "training_summary.json"
    if summary_file.exists():
        print("\nTraining Results:")
        try:
            with open(summary_file, 'r') as f:
                results = json.load(f)
            
            for model_type, metrics in results.items():
                if metrics and isinstance(metrics, dict):
                    print(f"\n{model_type.upper()} Model:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"   {metric}: {value:.4f}")
        except Exception as e:
            print(f"ERROR: Error reading training results: {e}")
    else:
        print("\nNo training summary found")
    
    print("\nQuick Commands:")
    print("   python start_api.py          # Start API server")
    print("   python test_api.py           # Test API endpoints") 

if __name__ == "__main__":
    check_models()
