#!/usr/bin/env python3
# setup_ml_project.py - Complete ML Project Setup Script

"""
Complete setup script for Fraud & Phishing Detection ML Models

This script will:
1. Create the project directory structure
2. Install dependencies
3. Collect sample datasets
4. Preprocess the data
5. Train all models
6. Test the API
7. Generate evaluation reports

Usage:
    python setup_ml_project.py [--skip-training] [--skip-data-collection]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time
import json
from datetime import datetime

def run_command(command, description="", check=True):
    """Run a command with error handling"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=False, text=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        return False

def create_directory_structure():
    """Create the complete ML project directory structure"""
    print("\n🏗️ Creating directory structure...")
    
    directories = [
        "ml_models",
        "ml_models/config",
        "ml_models/data",
        "ml_models/data/raw",
        "ml_models/data/raw/urls",
        "ml_models/data/raw/emails",
        "ml_models/data/raw/transactions",
        "ml_models/data/processed",
        "ml_models/data/processed/urls", 
        "ml_models/data/processed/emails",
        "ml_models/data/processed/transactions",
        "ml_models/data/collectors",
        "ml_models/preprocessing",
        "ml_models/models",
        "ml_models/training",
        "ml_models/inference", 
        "ml_models/saved",
        "ml_models/saved/models",
        "ml_models/saved/vectorizers",
        "ml_models/saved/scalers",
        "ml_models/evaluation",
        "ml_models/evaluation/reports",
        "ml_models/api",
        "ml_models/utils",
        "ml_models/tests",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith("ml_models") and "/" in directory:
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")
    
    print("✅ Directory structure created successfully!")

def create_requirements_file():
    """Create requirements.txt file"""
    print("\n📝 Creating requirements file...")
    
    requirements = """# Core ML libraries
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0

# Feature extraction
tldextract>=3.4.0
urllib3>=1.26.0

# API framework
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Data processing
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
python-dateutil>=2.8.0
requests>=2.31.0
"""
    
    req_file = Path("ml_models/requirements-ml.txt")
    req_file.write_text(requirements)
    print(f"✅ Requirements file created: {req_file}")

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Create requirements file if it doesn't exist
    req_file = Path("ml_models/requirements-ml.txt")
    if not req_file.exists():
        create_requirements_file()
    
    return run_command(
        f"pip install -r {req_file}",
        "Installing Python dependencies"
    )

def collect_sample_data():
    """Collect and generate sample datasets"""
    print("\n📊 Collecting sample datasets...")
    
    # Create a simple data collection script
    data_collector_script = """
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Generate sample URL data
def generate_url_data():
    legitimate_urls = [
        "https://www.google.com", "https://www.facebook.com", "https://www.amazon.com",
        "https://www.microsoft.com", "https://www.apple.com", "https://www.github.com",
        "https://stackoverflow.com", "https://www.wikipedia.org", "https://www.linkedin.com",
        "https://www.youtube.com", "https://www.twitter.com", "https://www.reddit.com"
    ]
    
    phishing_urls = [
        "http://paypal-security.tk/login", "https://amazon-account.ml/verify",
        "http://microsoft-update.cf/secure", "https://apple-id.ga/unlock",
        "http://bank-verify.tk/account", "https://facebook-security.ml/check",
        "http://google-account.cf/verify", "https://paypal-update.ga/login"
    ]
    
    df = pd.DataFrame({
        'url': legitimate_urls + phishing_urls,
        'label': ['legitimate'] * len(legitimate_urls) + ['phishing'] * len(phishing_urls)
    })
    
    df.to_csv('ml_models/data/raw/urls/raw_urls.csv', index=False)
    print(f"Generated {len(df)} URL samples")

# Generate sample email data
def generate_email_data():
    legitimate_emails = [
        "Thank you for your order. Your package will arrive soon.",
        "Your monthly statement is now available online.",
        "Meeting scheduled for tomorrow at 2 PM.",
        "Welcome to our newsletter! Here are this week's updates.",
        "Your subscription has been renewed successfully.",
        "Invoice attached for your recent purchase."
    ]
    
    phishing_emails = [
        "URGENT! Your account suspended. Click here to verify immediately.",
        "You won $1,000,000! Send processing fee to claim prize.",
        "Your bank account compromised. Update details now.",
        "ALERT: Suspicious activity detected. Verify identity now!",
        "Congratulations! You've been selected for a special offer.",
        "Action required: Confirm your payment information immediately."
    ]
    
    df = pd.DataFrame({
        'email_text': legitimate_emails + phishing_emails,
        'sender': ['orders@company.com'] * len(legitimate_emails) + ['security@fake.com'] * len(phishing_emails),
        'label': ['legitimate'] * len(legitimate_emails) + ['phishing'] * len(phishing_emails)
    })
    
    df.to_csv('ml_models/data/raw/emails/raw_emails.csv', index=False)
    print(f"Generated {len(df)} email samples")

# Generate sample transaction data  
def generate_transaction_data():
    np.random.seed(42)
    n_samples = 1000
    
    transactions = []
    for i in range(n_samples):
        is_fraud = random.random() < 0.1  # 10% fraud rate
        
        if is_fraud:
            amount = random.uniform(1000, 5000)  # High amounts for fraud
            hour = random.randint(0, 23)  # Any time
            merchant_category = random.choice(['online', 'atm', 'gas_station'])
        else:
            amount = random.uniform(10, 500)  # Normal amounts
            hour = random.choices(range(24), weights=[1]*6 + [3]*12 + [2]*6)[0]  # Business hours bias
            merchant_category = random.choice(['grocery', 'restaurant', 'retail'])
        
        transactions.append({
            'transaction_id': f'txn_{i}',
            'user_id': f'user_{i%100}',
            'amount': round(amount, 2),
            'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).replace(hour=hour).isoformat(),
            'merchant_category': merchant_category,
            'merchant_name': f'Merchant_{i%50}',
            'location': random.choice(['New York', 'Los Angeles', 'Chicago', '']),
            'label': 'fraud' if is_fraud else 'legitimate'
        })
    
    df = pd.DataFrame(transactions)
    df.to_csv('ml_models/data/raw/transactions/raw_transactions.csv', index=False)
    print(f"Generated {len(df)} transaction samples")

if __name__ == "__main__":
    generate_url_data()
    generate_email_data() 
    generate_transaction_data()
    print("Sample data generation complete!")
"""
    
    # Write and run the data collection script
    script_path = Path("generate_sample_data.py")
    script_path.write_text(data_collector_script)
    
    success = run_command("python generate_sample_data.py", "Generating sample datasets")
    
    # Clean up
    script_path.unlink()
    
    return success

def preprocess_data():
    """Preprocess the collected data"""
    print("\n🔧 Preprocessing data...")
    
    preprocessing_script = """
import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# Try to import tldextract, create fallback if not available
try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False
    print("Warning: tldextract not available, using basic URL parsing")

from urllib.parse import urlparse

def calculate_entropy(text):
    '''Calculate Shannon entropy of text'''
    if not text:
        return 0
    
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum(p * np.log2(p) for p in prob if p > 0)
    return entropy

# URL Feature Extraction
def extract_url_features(url):
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = len(re.findall(r'\\d', url))
    features['num_special_chars'] = len(re.findall(r'[^a-zA-Z0-9]', url))
    features['num_dots'] = url.count('.')
    features['is_https'] = 1 if url.startswith('https://') else 0
    
    try:
        if HAS_TLDEXTRACT:
            extracted = tldextract.extract(url)
            features['domain_length'] = len(extracted.domain)
            features['tld_length'] = len(extracted.suffix)
        else:
            parsed = urlparse(url)
            domain = parsed.netloc.split('.')[0] if '.' in parsed.netloc else parsed.netloc
            features['domain_length'] = len(domain)
            features['tld_length'] = len(parsed.netloc.split('.')[-1]) if '.' in parsed.netloc else 0
    except:
        features['domain_length'] = 0
        features['tld_length'] = 0
    
    features['has_ip'] = 1 if re.search(r'\\d+\\.\\d+\\.\\d+\\.\\d+', url) else 0
    features['has_suspicious_tld'] = 1 if any(tld in url.lower() for tld in ['.tk', '.ml', '.cf', '.ga']) else 0
    features['url_entropy'] = calculate_entropy(url)
    
    return features

def preprocess_urls():
    input_file = 'ml_models/data/raw/urls/raw_urls.csv'
    output_file = 'ml_models/data/processed/urls/processed_urls.csv'
    
    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    
    # Extract features
    features_list = []
    for _, row in df.iterrows():
        features = extract_url_features(row['url'])
        features['url'] = row['url']
        features['label'] = row['label']
        features_list.append(features)
    
    processed_df = pd.DataFrame(features_list)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_file, index=False)
    print(f"Processed URL data saved to {output_file}")

# Email Feature Extraction
def extract_email_features(email_text, sender=""):
    features = {}
    features['email_length'] = len(email_text)
    features['num_words'] = len(email_text.split())
    features['num_exclamation'] = email_text.count('!')
    features['num_caps'] = sum(1 for c in email_text if c.isupper())
    features['has_urgent'] = 1 if any(word in email_text.lower() for word in ['urgent', 'immediate', 'asap', 'alert']) else 0
    features['has_money'] = 1 if any(word in email_text.lower() for word in ['$', 'money', 'prize', 'win', 'million']) else 0
    features['has_click'] = 1 if any(word in email_text.lower() for word in ['click', 'link', 'here']) else 0
    features['sender_length'] = len(sender)
    features['has_suspicious_sender'] = 1 if any(word in sender.lower() for word in ['noreply', 'security', 'alert']) else 0
    features['caps_ratio'] = features['num_caps'] / len(email_text) if len(email_text) > 0 else 0
    return features

def preprocess_emails():
    input_file = 'ml_models/data/raw/emails/raw_emails.csv'
    output_file = 'ml_models/data/processed/emails/processed_emails.csv'
    
    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    
    # Extract features
    features_list = []
    for _, row in df.iterrows():
        features = extract_email_features(row['email_text'], row.get('sender', ''))
        features['email_text'] = row['email_text']
        features['label'] = row['label']
        features_list.append(features)
    
    processed_df = pd.DataFrame(features_list)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_file, index=False)
    print(f"Processed email data saved to {output_file}")

# Transaction Feature Extraction
def extract_transaction_features(row):
    features = {}
    features['amount'] = float(row['amount'])
    features['amount_log'] = np.log1p(features['amount'])
    
    # Parse timestamp
    try:
        from datetime import datetime
        timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
        features['hour'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        features['is_night'] = 1 if timestamp.hour < 6 or timestamp.hour > 22 else 0
    except:
        features['hour'] = 12
        features['day_of_week'] = 1  
        features['is_weekend'] = 0
        features['is_night'] = 0
    
    # Merchant features
    merchant_categories = {'grocery': 1, 'restaurant': 2, 'retail': 3, 'online': 4, 'atm': 5, 'gas_station': 6}
    features['merchant_category_encoded'] = merchant_categories.get(row.get('merchant_category', ''), 0)
    features['has_location'] = 1 if row.get('location', '') else 0
    
    # Amount-based features
    features['is_high_amount'] = 1 if features['amount'] > 1000 else 0
    features['is_round_amount'] = 1 if features['amount'] % 100 == 0 else 0
    
    return features

def preprocess_transactions():
    input_file = 'ml_models/data/raw/transactions/raw_transactions.csv'
    output_file = 'ml_models/data/processed/transactions/processed_transactions.csv'
    
    if not Path(input_file).exists():
        print(f"Input file not found: {input_file}")
        return
    
    df = pd.read_csv(input_file)
    
    # Extract features
    features_list = []
    for _, row in df.iterrows():
        features = extract_transaction_features(row)
        features['transaction_id'] = row['transaction_id']
        features['user_id'] = row['user_id']
        features['label'] = row['label']
        features_list.append(features)
    
    processed_df = pd.DataFrame(features_list)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_file, index=False)
    print(f"Processed transaction data saved to {output_file}")

if __name__ == "__main__":
    preprocess_urls()
    preprocess_emails()
    preprocess_transactions()
    print("Data preprocessing complete!")
"""
    
    # Write and run preprocessing script
    script_path = Path("preprocess_data.py")
    script_path.write_text(preprocessing_script)
    
    success = run_command("python preprocess_data.py", "Preprocessing datasets")
    
    # Clean up
    script_path.unlink()
    
    return success

def train_models():
    """Train all ML models"""
    print("\n🤖 Training ML models...")
    
    training_script = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def train_url_model():
    print("Training URL model...")
    
    # Load data
    data_path = 'ml_models/data/processed/urls/processed_urls.csv'
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    
    # Prepare features and labels
    feature_columns = [col for col in df.columns if col not in ['url', 'label']]
    X = df[feature_columns].fillna(0)
    y = (df['label'] == 'phishing').astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if len(set(y_test)) > 1 else [0.5] * len(y_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5
    }
    
    print("URL Model Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model and scaler
    Path('ml_models/saved/models').mkdir(parents=True, exist_ok=True)
    Path('ml_models/saved/scalers').mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, 'ml_models/saved/models/url_classifier.pkl')
    joblib.dump(scaler, 'ml_models/saved/scalers/url_scaler.pkl')
    
    return metrics

def train_email_model():
    print("Training Email model...")
    
    # Load data
    data_path = 'ml_models/data/processed/emails/processed_emails.csv'
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    
    # Prepare TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_features = vectorizer.fit_transform(df['email_text']).toarray()
    
    # Manual features
    feature_columns = [col for col in df.columns if col not in ['email_text', 'label']]
    manual_features = df[feature_columns].fillna(0).values
    
    # Combine features
    X = np.hstack([tfidf_features, manual_features])
    y = (df['label'] == 'phishing').astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if len(set(y_test)) > 1 else [0.5] * len(y_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5
    }
    
    print("Email Model Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model and vectorizer
    Path('ml_models/saved/models').mkdir(parents=True, exist_ok=True)
    Path('ml_models/saved/vectorizers').mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, 'ml_models/saved/models/email_classifier.pkl')
    joblib.dump(vectorizer, 'ml_models/saved/vectorizers/email_vectorizer.pkl')
    
    return metrics

def train_transaction_model():
    print("Training Transaction model...")
    
    # Load data
    data_path = 'ml_models/data/processed/transactions/processed_transactions.csv'
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    
    # Prepare features and labels  
    feature_columns = [col for col in df.columns if col not in ['transaction_id', 'user_id', 'label']]
    X = df[feature_columns].fillna(0)
    y = (df['label'] == 'fraud').astype(int)
    
    # Check if we have both classes
    if len(set(y)) < 2:
        print("Warning: Only one class present in transaction data")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with class balancing
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("Transaction Model Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model and scaler
    joblib.dump(model, 'ml_models/saved/models/transaction_classifier.pkl')
    joblib.dump(scaler, 'ml_models/saved/scalers/transaction_scaler.pkl')
    
    return metrics

if __name__ == "__main__":
    all_metrics = {}
    
    try:
        all_metrics['url'] = train_url_model()
    except Exception as e:
        print(f"Error training URL model: {e}")
        all_metrics['url'] = None
        
    try:
        all_metrics['email'] = train_email_model()
    except Exception as e:
        print(f"Error training Email model: {e}")
        all_metrics['email'] = None
        
    try:
        all_metrics['transaction'] = train_transaction_model()
    except Exception as e:
        print(f"Error training Transaction model: {e}")
        all_metrics['transaction'] = None
    
    # Save training summary
    import json
    Path('ml_models/evaluation/reports').mkdir(parents=True, exist_ok=True)
    with open('ml_models/evaluation/reports/training_summary.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print("\\nModel training complete! Check ml_models/evaluation/reports/training_summary.json for results.")
"""
    
    # Write and run training script
    script_path = Path("train_models.py")
    script_path.write_text(training_script)
    
    success = run_command("python train_models.py", "Training ML models")
    
    # Clean up
    script_path.unlink()
    
    return success

def create_api_files():
    """Create API endpoint files"""
    print("\n🔌 Creating API files...")
    
    # Create main API file - fixed version
    api_code = """from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List
import re
from urllib.parse import urlparse
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Try to import tldextract
try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False

app = FastAPI(
    title="Fraud & Phishing Detection API",
    description="ML-powered API for detecting fraud and phishing attempts",
    version="1.0.0"
)

# Pydantic models for requests
class URLCheck(BaseModel):
    url: str

class EmailCheck(BaseModel):
    email_text: str
    sender_email: str = ""

class TransactionCheck(BaseModel):
    user_id: str
    amount: float
    merchant_category: str
    merchant_name: str = ""
    location: str = ""

# Global model storage
models = {}
scalers = {}
vectorizers = {}

def calculate_entropy(text):
    '''Calculate Shannon entropy of text'''
    if not text:
        return 0
    
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum(p * np.log2(p) for p in prob if p > 0)
    return entropy

def load_models():
    '''Load all trained models'''
    model_path = Path("ml_models/saved/models")
    scaler_path = Path("ml_models/saved/scalers")
    vectorizer_path = Path("ml_models/saved/vectorizers")
    
    try:
        # Load URL model
        if (model_path / "url_classifier.pkl").exists():
            models['url'] = joblib.load(model_path / "url_classifier.pkl")
            scalers['url'] = joblib.load(scaler_path / "url_scaler.pkl")
            
        # Load Email model  
        if (model_path / "email_classifier.pkl").exists():
            models['email'] = joblib.load(model_path / "email_classifier.pkl")
            vectorizers['email'] = joblib.load(vectorizer_path / "email_vectorizer.pkl")
            
        # Load Transaction model
        if (model_path / "transaction_classifier.pkl").exists():
            models['transaction'] = joblib.load(model_path / "transaction_classifier.pkl")
            scalers['transaction'] = joblib.load(scaler_path / "transaction_scaler.pkl")
    except Exception as e:
        logging.error(f"Error loading models: {e}")

def extract_url_features(url: str):
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = len(re.findall(r'\\d', url))
    features['num_special_chars'] = len(re.findall(r'[^a-zA-Z0-9]', url))
    features['num_dots'] = url.count('.')
    features['is_https'] = 1 if url.startswith('https://') else 0
    
    try:
        if HAS_TLDEXTRACT:
            extracted = tldextract.extract(url)
            features['domain_length'] = len(extracted.domain)
            features['tld_length'] = len(extracted.suffix)
        else:
            parsed = urlparse(url)
            netloc = parsed.netloc
            if netloc:
                parts = netloc.split('.')
                features['domain_length'] = len(parts[0]) if len(parts) > 1 else len(netloc)
                features['tld_length'] = len(parts[-1]) if len(parts) > 1 else 0
            else:
                features['domain_length'] = 0
                features['tld_length'] = 0
    except:
        features['domain_length'] = 0
        features['tld_length'] = 0
    
    features['has_ip'] = 1 if re.search(r'\\d+\\.\\d+\\.\\d+\\.\\d+', url) else 0
    features['has_suspicious_tld'] = 1 if any(tld in url.lower() for tld in ['.tk', '.ml', '.cf', '.ga']) else 0
    features['url_entropy'] = calculate_entropy(url)
    
    return features

def extract_email_features(email_text: str, sender_email: str = ""):
    features = {}
    features['email_length'] = len(email_text)
    features['num_words'] = len(email_text.split())
    features['num_exclamation'] = email_text.count('!')
    features['num_caps'] = sum(1 for c in email_text if c.isupper())
    features['has_urgent'] = 1 if any(word in email_text.lower() for word in ['urgent', 'immediate', 'asap', 'alert']) else 0
    features['has_money'] = 1 if any(word in email_text.lower() for word in ['$', 'money', 'prize', 'win', 'million']) else 0
    features['has_click'] = 1 if any(word in email_text.lower() for word in ['click', 'link', 'here']) else 0
    features['sender_length'] = len(sender_email)
    features['has_suspicious_sender'] = 1 if any(word in sender_email.lower() for word in ['noreply', 'security', 'alert']) else 0
    features['caps_ratio'] = features['num_caps'] / features['email_length'] if features['email_length'] > 0 else 0
    return features

def extract_transaction_features(data: TransactionCheck):
    features = {}
    features['amount'] = data.amount
    features['amount_log'] = np.log1p(data.amount)
    
    now = datetime.now()
    features['hour'] = now.hour
    features['day_of_week'] = now.weekday()
    features['is_weekend'] = 1 if now.weekday() >= 5 else 0
    features['is_night'] = 1 if now.hour < 6 or now.hour > 22 else 0
    
    merchant_categories = {'grocery': 1, 'restaurant': 2, 'retail': 3, 'online': 4, 'atm': 5, 'gas_station': 6}
    features['merchant_category_encoded'] = merchant_categories.get(data.merchant_category.lower(), 0)
    features['has_location'] = 1 if data.location else 0
    
    features['is_high_amount'] = 1 if data.amount > 1000 else 0
    features['is_round_amount'] = 1 if data.amount % 100 == 0 else 0
    
    return features

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "models_loaded": list(models.keys())
    }

@app.post("/api/url-check")
def check_url(data: URLCheck):
    if 'url' not in models:
        raise HTTPException(status_code=503, detail="URL model not available")
    
    try:
        features = extract_url_features(data.url)
        feature_order = ['url_length', 'num_digits', 'num_special_chars', 'num_dots', 'is_https', 
                         'domain_length', 'tld_length', 'has_ip', 'has_suspicious_tld', 'url_entropy']
        feature_array = [features.get(f, 0) for f in feature_order]
        
        scaled = scalers['url'].transform([feature_array])
        pred = models['url'].predict(scaled)[0]
        proba = models['url'].predict_proba(scaled)[0][1]
        
        risk_level = "high" if proba > 0.8 else "medium" if proba > 0.4 else "low"
        
        return {
            "url": data.url,
            "is_phishing": pred == 1,
            "risk_level": risk_level,
            "confidence": round(proba, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/url-check/batch")
def check_url_batch(data: List[str]):
    if 'url' not in models:
        raise HTTPException(status_code=503, detail="URL model not available")
    
    results = []
    for url in data:
        try:
            features = extract_url_features(url)
            feature_order = ['url_length', 'num_digits', 'num_special_chars', 'num_dots', 'is_https', 
                             'domain_length', 'tld_length', 'has_ip', 'has_suspicious_tld', 'url_entropy']
            feature_array = [features.get(f, 0) for f in feature_order]
            
            scaled = scalers['url'].transform([feature_array])
            pred = models['url'].predict(scaled)[0]
            proba = models['url'].predict_proba(scaled)[0][1]
            
            risk_level = "high" if proba > 0.8 else "medium" if proba > 0.4 else "low"
            
            results.append({
                "url": url,
                "is_phishing": pred == 1,
                "risk_level": risk_level,
                "confidence": round(proba, 4)
            })
        except Exception as e:
            results.append({
                "url": url,
                "error": str(e)
            })
    return {"results": results}

@app.post("/api/email-check")
def check_email(data: EmailCheck):
    if 'email' not in models:
        raise HTTPException(status_code=503, detail="Email model not available")
    
    try:
        tfidf = vectorizers['email'].transform([data.email_text]).toarray()[0]
        
        features = extract_email_features(data.email_text, data.sender_email)
        feature_order = ['email_length', 'num_words', 'num_exclamation', 'num_caps', 'has_urgent', 
                         'has_money', 'has_click', 'sender_length', 'has_suspicious_sender', 'caps_ratio']
        manual_array = [features.get(f, 0) for f in feature_order]
        
        X = np.hstack([tfidf, manual_array])
        
        pred = models['email'].predict([X])[0]
        proba = models['email'].predict_proba([X])[0][1]
        
        risk_level = "high" if proba > 0.8 else "medium" if proba > 0.4 else "low"
        
        return {
            "email_text": data.email_text,
            "sender_email": data.sender_email,
            "is_phishing": pred == 1,
            "risk_level": risk_level,
            "confidence": round(proba, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transaction-check")
def check_transaction(data: TransactionCheck):
    if 'transaction' not in models:
        raise HTTPException(status_code=503, detail="Transaction model not available")
    
    try:
        features = extract_transaction_features(data)
        feature_order = ['amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend', 
                         'is_night', 'merchant_category_encoded', 'has_location', 
                         'is_high_amount', 'is_round_amount']
        feature_array = [features.get(f, 0) for f in feature_order]
        
        scaled = scalers['transaction'].transform([feature_array])
        pred = models['transaction'].predict(scaled)[0]
        proba = models['transaction'].predict_proba(scaled)[0][1]
        
        risk_level = "high" if proba > 0.8 else "medium" if proba > 0.4 else "low"
        
        return {
            "user_id": data.user_id,
            "amount": data.amount,
            "merchant_category": data.merchant_category,
            "is_fraud": pred == 1,
            "risk_level": risk_level,
            "confidence": round(proba, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
    
    # Write API file
    api_path = Path("ml_models/api/endpoints.py")
    api_path.parent.mkdir(parents=True, exist_ok=True)
    api_path.write_text(api_code)
    
    # Create __init__.py for api package
    (api_path.parent / "__init__.py").write_text("")
    
    print("✅ API files created successfully!")

def test_api():
    """Test the API endpoints"""
    print("\n🧪 Creating API test script...")
    
    api_test_script = """
import requests
import json
import time
from datetime import datetime

def test_api_endpoints():
    base_url = "http://localhost:8000"
    
    print("Testing API endpoints...")
    print("=" * 50)
    
    # Test health endpoint
    print("\\n1. Testing Health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        print("   Make sure the API server is running!")
        return
    
    # Test URL check
    print("\\n2. Testing URL check endpoint...")
    try:
        test_data = {"url": "https://paypal-security.tk/login"}
        response = requests.post(f"{base_url}/api/url-check", json=test_data, timeout=10)
        if response.status_code == 200:
            print("✅ URL check endpoint working")
            result = response.json()
            print(f"   URL: {result['url']}")
            print(f"   Is Phishing: {result['is_phishing']}")
            print(f"   Risk Level: {result['risk_level']}")
        else:
            print(f"❌ URL check failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ URL check error: {e}")
    
    # Test email check
    print("\\n3. Testing Email check endpoint...")
    try:
        test_data = {
            "email_text": "URGENT! Your account suspended. Click here to verify immediately.",
            "sender_email": "security@fake.com"
        }
        response = requests.post(f"{base_url}/api/email-check", json=test_data, timeout=10)
        if response.status_code == 200:
            print("✅ Email check endpoint working")
            result = response.json()
            print(f"   Is Phishing: {result['is_phishing']}")
            print(f"   Risk Level: {result['risk_level']}")
        else:
            print(f"❌ Email check failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Email check error: {e}")
    
    # Test transaction check
    print("\\n4. Testing Transaction check endpoint...")
    try:
        test_data = {
            "user_id": "test_user",
            "amount": 2500.0,
            "merchant_category": "atm",
            "merchant_name": "ATM Downtown",
            "location": "Unknown"
        }
        response = requests.post(f"{base_url}/api/transaction-check", json=test_data, timeout=10)
        if response.status_code == 200:
            print("✅ Transaction check endpoint working")
            result = response.json()
            print(f"   Amount: ${result['amount']}")
            print(f"   Is Fraud: {result['is_fraud']}")
            print(f"   Risk Level: {result['risk_level']}")
        else:
            print(f"❌ Transaction check failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Transaction check error: {e}")
    
    # Test batch URL check
    print("\\n5. Testing Batch URL check...")
    try:
        test_urls = [
            "https://www.google.com",
            "https://paypal-security.tk/login",
            "https://amazon-account.ml/verify"
        ]
        response = requests.post(f"{base_url}/api/url-check/batch", json=test_urls, timeout=15)
        if response.status_code == 200:
            print("✅ Batch URL check working")
            results = response.json()['results']
            for i, result in enumerate(results):
                if 'error' not in result:
                    print(f"   URL {i+1}: {result['risk_level']} risk")
                else:
                    print(f"   URL {i+1}: Error - {result['error']}")
        else:
            print(f"❌ Batch URL check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Batch URL check error: {e}")
    
    print("\\n" + "=" * 50)
    print("API testing complete!")
    print("\\n📖 For full API documentation, visit: http://localhost:8000/docs")

if __name__ == "__main__":
    print("🧪 Fraud & Phishing Detection API - Test Suite")
    print("=" * 50)
    print("📝 Make sure the API server is running:")
    print("   python start_api.py")
    print("\\nStarting tests in 3 seconds...")
    time.sleep(3)
    test_api_endpoints()
"""
    
    # Write test script
    script_path = Path("test_api.py")
    script_path.write_text(api_test_script)
    
    print("✅ API test script created (test_api.py)")
    
    return True

def create_startup_scripts():
    """Create convenient startup scripts"""
    print("\n📝 Creating startup scripts...")
    
    # API startup script
    api_script = """#!/usr/bin/env python3
# start_api.py - Start the ML API server

import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("🚀 Starting Fraud & Phishing Detection API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("⚡ Interactive API: http://localhost:8000/redoc")
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
        print(f"❌ Import error: {e}")
        print("📥 Please install dependencies: pip install -r ml_models/requirements-ml.txt")
    except Exception as e:
        print(f"❌ Error starting API: {e}")
"""
    
    Path("start_api.py").write_text(api_script)
    
    # Model checker script
    check_script = """#!/usr/bin/env python3
# check_models.py - Check model status and performance

import json
from pathlib import Path
import joblib

def check_models():
    print("🔍 Checking ML Models Status")
    print("=" * 50)
    
    model_path = Path("ml_models/saved/models")
    reports_path = Path("ml_models/evaluation/reports")
    
    # Check if models exist
    models = {
        'url_classifier.pkl': 'URL Phishing Detection',
        'email_classifier.pkl': 'Email Phishing Detection', 
        'transaction_classifier.pkl': 'Transaction Fraud Detection'
    }
    
    print("\\n📊 Model Files:")
    for model_file, description in models.items():
        model_filepath = model_path / model_file
        if model_filepath.exists():
            size_mb = model_filepath.stat().st_size / (1024 * 1024)
            print(f"✅ {description}: {size_mb:.2f} MB")
        else:
            print(f"❌ {description}: Not found")
    
    # Check training summary
    summary_file = reports_path / "training_summary.json"
    if summary_file.exists():
        print("\\n📈 Training Results:")
        try:
            with open(summary_file) as f:
                results = json.load(f)
            
            for model_type, metrics in results.items():
                if metrics and isinstance(metrics, dict):
                    print(f"\\n🤖 {model_type.upper()} Model:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"   {metric}: {value:.4f}")
        except Exception as e:
            print(f"❌ Error reading training results: {e}")
    else:
        print("\\n📋 No training summary found")
    
    print("\\n🚀 Quick Commands:")
    print("   python start_api.py          # Start API server")
    print("   python test_api.py           # Test API endpoints") 
    print("   python retrain_models.py     # Retrain all models")

if __name__ == "__main__":
    check_models()
"""
    
    Path("check_models.py").write_text(check_script)
    
    # Simple retraining script
    retrain_script = """#!/usr/bin/env python3
# retrain_models.py - Simple model retraining

import subprocess
import sys
from pathlib import Path

def retrain_all():
    print("🔄 Retraining all ML models...")
    print("=" * 60)
    
    # Run the main setup with specific flags
    try:
        result = subprocess.run([
            sys.executable, "setup_ml_project.py", 
            "--skip-data-collection"
        ], check=True)
        
        print("\\n🎉 Model retraining complete!")
        print("🚀 Start the API: python start_api.py")
        print("🧪 Test the API: python test_api.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Retraining failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        return False
    
    return True

if __name__ == "__main__":
    retrain_all()
"""
    
    Path("retrain_models.py").write_text(retrain_script)
    
    print("✅ Startup scripts created:")
    print("   - start_api.py: Start the API server")
    print("   - retrain_models.py: Retrain all models") 
    print("   - check_models.py: Check model status")

def generate_project_summary():
    """Generate a project summary report"""
    print("\n📋 Generating project summary...")
    
    summary = {
        "project": "Fraud & Phishing Detection ML System",
        "created": datetime.now().isoformat(),
        "version": "1.0.0",
        "structure": {
            "models": ["URL Classifier", "Email Classifier", "Transaction Classifier"],
            "algorithms": ["Random Forest", "Logistic Regression", "Random Forest"],
            "features": {
                "url": ["Length, Dots, HTTPS, Domain analysis, Entropy, Suspicious TLDs"],
                "email": ["TF-IDF + Manual features (urgency, money terms, caps ratio)"],  
                "transaction": ["Amount, Time patterns, Merchant category, Behavioral features"]
            }
        },
        "api_endpoints": [
            "/health - Check API status",
            "/api/url-check - Single URL analysis",
            "/api/email-check - Single email analysis", 
            "/api/transaction-check - Single transaction analysis",
            "/api/*/batch - Batch processing endpoints"
        ],
        "files_created": [
            "ml_models/saved/models/*.pkl - Trained models",
            "ml_models/saved/scalers/*.pkl - Feature scalers", 
            "ml_models/saved/vectorizers/*.pkl - Text vectorizers",
            "ml_models/data/processed/*.csv - Processed datasets",
            "ml_models/api/endpoints.py - FastAPI application",
            "start_api.py - API server launcher",
            "retrain_models.py - Model retraining script",
            "test_api.py - API testing script",
            "check_models.py - Model status checker"
        ],
        "next_steps": [
            "1. Start API: python start_api.py",
            "2. Test API: python test_api.py", 
            "3. Check docs: http://localhost:8000/docs",
            "4. Check models: python check_models.py",
            "5. Retrain models: python retrain_models.py"
        ]
    }
    
    # Save summary
    with open("project_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create README.md
    readme_content = f"""# Fraud & Phishing Detection ML System

🚀 **Complete machine learning system for detecting fraud and phishing attempts**

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📋 Overview

This system provides real-time detection of:
- **Phishing URLs** - Malicious websites attempting to steal credentials
- **Phishing Emails** - Fraudulent emails designed to trick users
- **Fraudulent Transactions** - Suspicious financial transactions

## 🚀 Quick Start

### 1. Start the API Server
```bash
python start_api.py"""