#!/usr/bin/env python3
"""
Windows-Compatible Fraud & Phishing Detection ML Project Runner
Fixed encoding issues for Windows systems
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def create_startup_scripts():
    """Create convenience scripts for running the system - Windows compatible"""
    print("Creating startup scripts...")
    
    # API startup script (no emoji)
    api_script = '''#!/usr/bin/env python3
import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("Starting Fraud & Phishing Detection API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print()
    
    try:
        uvicorn.run(
            "ml_models.api.endpoints:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install dependencies: pip install fastapi uvicorn")
    except Exception as e:
        print(f"Error starting API: {e}")
'''
    
    # Write with explicit encoding
    with open("start_api.py", "w", encoding="utf-8") as f:
        f.write(api_script)
    
    # Test script (no emoji)
    test_script = '''#!/usr/bin/env python3
import requests
import json
import time
from datetime import datetime

def test_api():
    base_url = "http://localhost:8000"
    
    print("Testing Fraud & Phishing Detection API")
    print("=" * 50)
    
    # Test health
    print("\\n1. Testing Health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"FAILED: Health endpoint failed: {response.status_code}")
            return
    except Exception as e:
        print(f"ERROR: Health endpoint error: {e}")
        print("   Make sure the API server is running!")
        return
    
    # Test URL check
    print("\\n2. Testing URL check...")
    test_data = {"url": "https://paypal-security.tk/login"}
    try:
        response = requests.post(f"{base_url}/api/url-check", json=test_data, timeout=10)
        if response.status_code == 200:
            print("SUCCESS: URL check working")
            result = response.json()
            print(f"   URL: {result['url']}")
            print(f"   Is Phishing: {result['is_phishing']}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Confidence: {result['confidence']}")
        else:
            print(f"FAILED: URL check failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"ERROR: URL check error: {e}")
    
    # Test email check
    print("\\n3. Testing Email check...")
    test_data = {
        "email_text": "URGENT! Your account suspended. Click here immediately.",
        "sender_email": "security@fake.com"
    }
    try:
        response = requests.post(f"{base_url}/api/email-check", json=test_data, timeout=10)
        if response.status_code == 200:
            print("SUCCESS: Email check working")
            result = response.json()
            print(f"   Is Phishing: {result['is_phishing']}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Confidence: {result['confidence']}")
        else:
            print(f"FAILED: Email check failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"ERROR: Email check error: {e}")
    
    # Test transaction check
    print("\\n4. Testing Transaction check...")
    test_data = {
        "user_id": "test_user",
        "amount": 2500.0,
        "merchant_category": "atm",
        "location": "Unknown"
    }
    try:
        response = requests.post(f"{base_url}/api/transaction-check", json=test_data, timeout=10)
        if response.status_code == 200:
            print("SUCCESS: Transaction check working")
            result = response.json()
            print(f"   Is Fraud: {result['is_fraud']}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Confidence: {result['confidence']}")
        else:
            print(f"FAILED: Transaction check failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"ERROR: Transaction check error: {e}")
    
    # Test batch processing
    print("\\n5. Testing Batch URL check...")
    test_urls = [
        "https://www.google.com",
        "https://paypal-security.tk/login", 
        "https://amazon-account.ml/verify"
    ]
    try:
        response = requests.post(f"{base_url}/api/url-check/batch", json=test_urls, timeout=15)
        if response.status_code == 200:
            print("SUCCESS: Batch URL check working")
            results = response.json()['results']
            for i, result in enumerate(results):
                if 'error' not in result:
                    print(f"   URL {i+1}: {result['risk_level']} risk ({result['confidence']:.3f})")
                else:
                    print(f"   URL {i+1}: Error - {result['error']}")
        else:
            print(f"FAILED: Batch URL check failed: {response.status_code}")
    except Exception as e:
        print(f"ERROR: Batch URL check error: {e}")
    
    print("\\n" + "=" * 50)
    print("API testing complete!")

if __name__ == "__main__":
    print("Starting tests in 3 seconds...")
    time.sleep(3)
    test_api()
'''
    
    with open("test_api.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    # Model checker script
    check_script = '''#!/usr/bin/env python3
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
    
    print("\\nModel Files:")
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
        print("\\nTraining Results:")
        try:
            with open(summary_file, 'r') as f:
                results = json.load(f)
            
            for model_type, metrics in results.items():
                if metrics and isinstance(metrics, dict):
                    print(f"\\n{model_type.upper()} Model:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"   {metric}: {value:.4f}")
        except Exception as e:
            print(f"ERROR: Error reading training results: {e}")
    else:
        print("\\nNo training summary found")
    
    print("\\nQuick Commands:")
    print("   python start_api.py          # Start API server")
    print("   python test_api.py           # Test API endpoints") 

if __name__ == "__main__":
    check_models()
'''
    
    with open("check_models.py", "w", encoding="utf-8") as f:
        f.write(check_script)
    
    print("Startup scripts created successfully!")

def generate_project_summary():
    """Generate a project summary report"""
    print("Generating project summary...")
    
    summary = {
        "project": "Fraud & Phishing Detection ML System",
        "created": datetime.now().isoformat(),
        "version": "1.0.0",
        "status": "Setup Complete",
        "models": {
            "url_classifier": "Random Forest - Phishing URL Detection",
            "email_classifier": "Logistic Regression + TF-IDF - Email Phishing Detection",  
            "transaction_classifier": "Random Forest - Transaction Fraud Detection"
        },
        "api_endpoints": [
            "/health - Check API status",
            "/api/url-check - Single URL analysis",
            "/api/email-check - Single email analysis", 
            "/api/transaction-check - Single transaction analysis",
            "/api/url-check/batch - Batch URL processing"
        ],
        "files_created": [
            "ml_models/saved/models/*.pkl - Trained models",
            "ml_models/saved/scalers/*.pkl - Feature scalers", 
            "ml_models/saved/vectorizers/*.pkl - Text vectorizers",
            "ml_models/data/processed/*.csv - Processed datasets",
            "ml_models/api/endpoints.py - FastAPI application",
            "start_api.py - API server launcher",
            "test_api.py - API testing script",
            "check_models.py - Model status checker"
        ]
    }
    
    # Save summary
    with open("project_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print("Project summary created successfully!")

def print_final_results():
    """Print final setup results"""
    print("\\n" + "=" * 60)
    print("PROJECT SETUP COMPLETE!")
    print("=" * 60)
    
    print("\\nModel Performance Summary:")
    try:
        with open('ml_models/evaluation/reports/training_summary.json', 'r') as f:
            results = json.load(f)
        
        for model_type, metrics in results.items():
            if metrics and isinstance(metrics, dict):
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1_score', 0)
                print(f"{model_type.upper()} Model - Accuracy: {acc:.3f}, F1: {f1:.3f}")
    except:
        print("Could not load training results")
    
    print("\\nNext Steps:")
    print("1. Start the API server:")
    print("   python start_api.py")
    print()
    print("2. In another terminal, test the API:")
    print("   python test_api.py")
    print()
    print("3. Access API documentation:")
    print("   http://localhost:8000/docs")
    print()
    print("4. Check model status:")
    print("   python check_models.py")
    
    print("\\nProject Structure:")
    print("- ml_models/saved/models/ - Trained ML models")
    print("- ml_models/data/ - Training datasets") 
    print("- ml_models/api/ - FastAPI application")
    print("- start_api.py - API server launcher")
    print("- test_api.py - API testing script")
    print("- check_models.py - Model status checker")

if __name__ == "__main__":
    print("Completing project setup...")
    
    try:
        create_startup_scripts()
        generate_project_summary()
        print_final_results()
        print("\\nSUCCESS: Setup completed successfully!")
        print("Run 'python start_api.py' to start the API server.")
        
    except Exception as e:
        print(f"ERROR: Final setup steps failed: {e}")
        print("However, your models are trained and ready to use!")