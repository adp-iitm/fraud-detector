#!/usr/bin/env python3
"""
Complete Fraud & Phishing Detection ML Project Runner
This script sets up and runs the entire ML system
"""

import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import joblib
    print("✅ Core ML libraries loaded successfully")
except ImportError as e:
    print(f"❌ Missing ML libraries: {e}")
    print("Please install: pip install scikit-learn pandas numpy joblib")
    sys.exit(1)

def create_directory_structure():
    """Create the complete ML project directory structure"""
    print("\n🗂️ Creating directory structure...")
    
    directories = [
        "ml_models",
        "ml_models/config",
        "ml_models/data",
        "ml_models/data/raw/urls",
        "ml_models/data/raw/emails",
        "ml_models/data/raw/transactions",
        "ml_models/data/processed/urls", 
        "ml_models/data/processed/emails",
        "ml_models/data/processed/transactions",
        "ml_models/data/collectors",
        "ml_models/preprocessing",
        "ml_models/models",
        "ml_models/training",
        "ml_models/inference", 
        "ml_models/saved/models",
        "ml_models/saved/vectorizers",
        "ml_models/saved/scalers",
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
            init_file.write_text("")
    
    print("✅ Directory structure created successfully!")

def generate_sample_data():
    """Generate sample datasets for training"""
    print("\n📊 Generating sample datasets...")
    
    # Generate sample URL data
    legitimate_urls = [
        "https://www.google.com", "https://www.facebook.com", "https://www.amazon.com",
        "https://www.microsoft.com", "https://www.apple.com", "https://www.github.com",
        "https://stackoverflow.com", "https://www.wikipedia.org", "https://www.linkedin.com",
        "https://www.youtube.com", "https://www.twitter.com", "https://www.reddit.com",
        "https://www.netflix.com", "https://www.spotify.com", "https://www.dropbox.com"
    ]
    
    phishing_urls = [
        "http://paypal-security.tk/login", "https://amazon-account.ml/verify",
        "http://microsoft-update.cf/secure", "https://apple-id.ga/unlock",
        "http://bank-verify.tk/account", "https://facebook-security.ml/check",
        "http://google-account.cf/verify", "https://paypal-update.ga/login",
        "http://netflix-billing.tk/update", "https://spotify-premium.cf/claim"
    ]
    
    url_df = pd.DataFrame({
        'url': legitimate_urls + phishing_urls,
        'label': ['legitimate'] * len(legitimate_urls) + ['phishing'] * len(phishing_urls)
    })
    
    url_df.to_csv('ml_models/data/raw/urls/raw_urls.csv', index=False)
    print(f"Generated {len(url_df)} URL samples")
    
    # Generate sample email data
    legitimate_emails = [
        "Thank you for your order. Your package will arrive soon.",
        "Your monthly statement is now available online.",
        "Meeting scheduled for tomorrow at 2 PM.",
        "Welcome to our newsletter! Here are this week's updates.",
        "Your subscription has been renewed successfully.",
        "Invoice attached for your recent purchase.",
        "Your account balance is $1,234.56 as of today.",
        "Password changed successfully for your account."
    ]
    
    phishing_emails = [
        "URGENT! Your account suspended. Click here to verify immediately.",
        "You won $1,000,000! Send processing fee to claim prize.",
        "Your bank account compromised. Update details now.",
        "ALERT: Suspicious activity detected. Verify identity now!",
        "Congratulations! You've been selected for a special offer.",
        "Action required: Confirm your payment information immediately.",
        "FINAL NOTICE: Account will be closed unless you click here.",
        "Security breach detected! Update your password NOW!"
    ]
    
    email_df = pd.DataFrame({
        'email_text': legitimate_emails + phishing_emails,
        'sender': ['orders@company.com'] * len(legitimate_emails) + ['security@fake.com'] * len(phishing_emails),
        'label': ['legitimate'] * len(legitimate_emails) + ['phishing'] * len(phishing_emails)
    })
    
    email_df.to_csv('ml_models/data/raw/emails/raw_emails.csv', index=False)
    print(f"Generated {len(email_df)} email samples")
    
    # Generate sample transaction data  
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
    
    transaction_df = pd.DataFrame(transactions)
    transaction_df.to_csv('ml_models/data/raw/transactions/raw_transactions.csv', index=False)
    print(f"Generated {len(transaction_df)} transaction samples")

def calculate_entropy(text):
    """Calculate Shannon entropy of text"""
    if not text:
        return 0
    
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum(p * np.log2(p) for p in prob if p > 0)
    return entropy

def extract_url_features(url):
    """Extract features from URL"""
    import re
    from urllib.parse import urlparse
    
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = len(re.findall(r'\d', url))
    features['num_special_chars'] = len(re.findall(r'[^a-zA-Z0-9]', url))
    features['num_dots'] = url.count('.')
    features['is_https'] = 1 if url.startswith('https://') else 0
    
    try:
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
    
    features['has_ip'] = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    features['has_suspicious_tld'] = 1 if any(tld in url.lower() for tld in ['.tk', '.ml', '.cf', '.ga']) else 0
    features['url_entropy'] = calculate_entropy(url)
    
    return features

def extract_email_features(email_text, sender=""):
    """Extract features from email"""
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

def extract_transaction_features(row):
    """Extract features from transaction"""
    features = {}
    features['amount'] = float(row['amount'])
    features['amount_log'] = np.log1p(features['amount'])
    
    # Parse timestamp
    try:
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

def preprocess_data():
    """Preprocess all datasets"""
    print("\n🔧 Preprocessing data...")
    
    # Process URLs
    print("Processing URLs...")
    url_df = pd.read_csv('ml_models/data/raw/urls/raw_urls.csv')
    
    url_features_list = []
    for _, row in url_df.iterrows():
        features = extract_url_features(row['url'])
        features['url'] = row['url']
        features['label'] = row['label']
        url_features_list.append(features)
    
    url_processed_df = pd.DataFrame(url_features_list)
    url_processed_df.to_csv('ml_models/data/processed/urls/processed_urls.csv', index=False)
    
    # Process Emails
    print("Processing emails...")
    email_df = pd.read_csv('ml_models/data/raw/emails/raw_emails.csv')
    
    email_features_list = []
    for _, row in email_df.iterrows():
        features = extract_email_features(row['email_text'], row.get('sender', ''))
        features['email_text'] = row['email_text']
        features['label'] = row['label']
        email_features_list.append(features)
    
    email_processed_df = pd.DataFrame(email_features_list)
    email_processed_df.to_csv('ml_models/data/processed/emails/processed_emails.csv', index=False)
    
    # Process Transactions
    print("Processing transactions...")
    transaction_df = pd.read_csv('ml_models/data/raw/transactions/raw_transactions.csv')
    
    transaction_features_list = []
    for _, row in transaction_df.iterrows():
        features = extract_transaction_features(row)
        features['transaction_id'] = row['transaction_id']
        features['user_id'] = row['user_id']
        features['label'] = row['label']
        transaction_features_list.append(features)
    
    transaction_processed_df = pd.DataFrame(transaction_features_list)
    transaction_processed_df.to_csv('ml_models/data/processed/transactions/processed_transactions.csv', index=False)
    
    print("✅ Data preprocessing complete!")

def train_models():
    """Train all ML models"""
    print("\n🤖 Training ML models...")
    
    all_metrics = {}
    
    # Train URL model
    print("Training URL model...")
    try:
        df = pd.read_csv('ml_models/data/processed/urls/processed_urls.csv')
        feature_columns = [col for col in df.columns if col not in ['url', 'label']]
        X = df[feature_columns].fillna(0)
        y = (df['label'] == 'phishing').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if len(set(y_test)) > 1 else [0.5] * len(y_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5
        }
        
        all_metrics['url'] = metrics
        joblib.dump(model, 'ml_models/saved/models/url_classifier.pkl')
        joblib.dump(scaler, 'ml_models/saved/scalers/url_scaler.pkl')
        
        print("URL Model Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        print(f"Error training URL model: {e}")
        all_metrics['url'] = None
    
    # Train Email model
    print("Training Email model...")
    try:
        df = pd.read_csv('ml_models/data/processed/emails/processed_emails.csv')
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_features = vectorizer.fit_transform(df['email_text']).toarray()
        
        feature_columns = [col for col in df.columns if col not in ['email_text', 'label']]
        manual_features = df[feature_columns].fillna(0).values
        
        X = np.hstack([tfidf_features, manual_features])
        y = (df['label'] == 'phishing').astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if len(set(y_test)) > 1 else [0.5] * len(y_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5
        }
        
        all_metrics['email'] = metrics
        joblib.dump(model, 'ml_models/saved/models/email_classifier.pkl')
        joblib.dump(vectorizer, 'ml_models/saved/vectorizers/email_vectorizer.pkl')
        
        print("Email Model Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
    except Exception as e:
        print(f"Error training Email model: {e}")
        all_metrics['email'] = None
    
    # Train Transaction model
    print("Training Transaction model...")
    try:
        df = pd.read_csv('ml_models/data/processed/transactions/processed_transactions.csv')
        feature_columns = [col for col in df.columns if col not in ['transaction_id', 'user_id', 'label']]
        X = df[feature_columns].fillna(0)
        y = (df['label'] == 'fraud').astype(int)
        
        if len(set(y)) < 2:
            print("Warning: Only one class present in transaction data")
            all_metrics['transaction'] = None
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            all_metrics['transaction'] = metrics
            joblib.dump(model, 'ml_models/saved/models/transaction_classifier.pkl')
            joblib.dump(scaler, 'ml_models/saved/scalers/transaction_scaler.pkl')
            
            print("Transaction Model Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
                
    except Exception as e:
        print(f"Error training Transaction model: {e}")
        all_metrics['transaction'] = None
    
    # Save training summary
    with open('ml_models/evaluation/reports/training_summary.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print("\n✅ Model training complete!")
    return all_metrics

def create_api():
    """Create FastAPI application"""
    print("\n🔌 Creating API application...")
    
    api_code = '''from fastapi import FastAPI, HTTPException
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
    if not text:
        return 0
    
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum(p * np.log2(p) for p in prob if p > 0)
    return entropy

def load_models():
    model_path = Path("ml_models/saved/models")
    scaler_path = Path("ml_models/saved/scalers")
    vectorizer_path = Path("ml_models/saved/vectorizers")
    
    try:
        if (model_path / "url_classifier.pkl").exists():
            models['url'] = joblib.load(model_path / "url_classifier.pkl")
            scalers['url'] = joblib.load(scaler_path / "url_scaler.pkl")
            
        if (model_path / "email_classifier.pkl").exists():
            models['email'] = joblib.load(model_path / "email_classifier.pkl")
            vectorizers['email'] = joblib.load(vectorizer_path / "email_vectorizer.pkl")
            
        if (model_path / "transaction_classifier.pkl").exists():
            models['transaction'] = joblib.load(model_path / "transaction_classifier.pkl")
            scalers['transaction'] = joblib.load(scaler_path / "transaction_scaler.pkl")
    except Exception as e:
        logging.error(f"Error loading models: {e}")

def extract_url_features(url: str):
    features = {}
    features['url_length'] = len(url)
    features['num_digits'] = len(re.findall(r'\\\\d', url))
    features['num_special_chars'] = len(re.findall(r'[^a-zA-Z0-9]', url))
    features['num_dots'] = url.count('.')
    features['is_https'] = 1 if url.startswith('https://') else 0
    
    try:
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
    
    features['has_ip'] = 1 if re.search(r'\\\\d+\\\\.\\\\d+\\\\.\\\\d+\\\\.\\\\d+', url) else 0
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

@app.post("/api/url-check/batch")
def check_url_batch(urls: List[str]):
    if 'url' not in models:
        raise HTTPException(status_code=503, detail="URL model not available")
    
    results = []
    for url in urls:
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
'''
    
    # Write API file
    api_path = Path("ml_models/api/endpoints.py")
    api_path.parent.mkdir(parents=True, exist_ok=True)
    api_path.write_text(api_code)
    
    # Create __init__.py for api package
    (api_path.parent / "__init__.py").write_text("")
    
    print("API application created successfully!")

def create_startup_scripts():
    """Create convenience scripts for running the system"""
    print("Creating startup scripts...")
    
    # API startup script
    api_script = '''#!/usr/bin/env python3
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
        print("🔥 Please install dependencies: pip install fastapi uvicorn")
    except Exception as e:
        print(f"❌ Error starting API: {e}")
'''
    
    Path("start_api.py").write_text(api_script)
    
    # Test script
    test_script = '''#!/usr/bin/env python3
import requests
import json
import time
from datetime import datetime

def test_api():
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Fraud & Phishing Detection API")
    print("=" * 50)
    
    # Test health
    print("\\n1. Testing Health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        print("   Make sure the API server is running!")
        return
    
    # Test URL check
    print("\\n2. Testing URL check...")
    test_data = {"url": "https://paypal-security.tk/login"}
    try:
        response = requests.post(f"{base_url}/api/url-check", json=test_data, timeout=10)
        if response.status_code == 200:
            print("✅ URL check working")
            result = response.json()
            print(f"   URL: {result['url']}")
            print(f"   Is Phishing: {result['is_phishing']}")
            print(f"   Risk Level: {result['risk_level']}")
        else:
            print(f"❌ URL check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ URL check error: {e}")
    
    # Test email check
    print("\\n3. Testing Email check...")
    test_data = {
        "email_text": "URGENT! Your account suspended. Click here immediately.",
        "sender_email": "security@fake.com"
    }
    try:
        response = requests.post(f"{base_url}/api/email-check", json=test_data, timeout=10)
        if response.status_code == 200:
            print("✅ Email check working")
            result = response.json()
            print(f"   Is Phishing: {result['is_phishing']}")
            print(f"   Risk Level: {result['risk_level']}")
        else:
            print(f"❌ Email check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Email check error: {e}")
    
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
            print("✅ Transaction check working")
            result = response.json()
            print(f"   Is Fraud: {result['is_fraud']}")
            print(f"   Risk Level: {result['risk_level']}")
        else:
            print(f"❌ Transaction check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Transaction check error: {e}")
    
    print("\\n" + "=" * 50)
    print("API testing complete!")

if __name__ == "__main__":
    print("Starting tests in 3 seconds...")
    time.sleep(3)
    test_api()
'''
    
    Path("test_api.py").write_text(test_script)
    
    print("Startup scripts created successfully!")

def run_complete_setup():
    """Run the complete ML project setup"""
    print("🚀 Fraud & Phishing Detection ML Project Setup")
    print("=" * 60)
    print("Setting up complete machine learning system for:")
    print("- Phishing URL detection")
    print("- Phishing email detection") 
    print("- Fraudulent transaction detection")
    print("=" * 60)
    
    try:
        # Step 1: Create directory structure
        create_directory_structure()
        
        # Step 2: Generate sample data
        generate_sample_data()
        
        # Step 3: Preprocess data
        preprocess_data()
        
        # Step 4: Train models
        metrics = train_models()
        
        # Step 5: Create API
        create_api()
        
        # Step 6: Create startup scripts
        create_startup_scripts()
        
        # Step 7: Generate final summary
        print("\n🎉 PROJECT SETUP COMPLETE!")
        print("=" * 60)
        
        print("\n📊 Model Performance Summary:")
        if metrics.get('url'):
            print(f"URL Model - Accuracy: {metrics['url']['accuracy']:.3f}, F1: {metrics['url']['f1_score']:.3f}")
        if metrics.get('email'):
            print(f"Email Model - Accuracy: {metrics['email']['accuracy']:.3f}, F1: {metrics['email']['f1_score']:.3f}")
        if metrics.get('transaction'):
            print(f"Transaction Model - Accuracy: {metrics['transaction']['accuracy']:.3f}, F1: {metrics['transaction']['f1_score']:.3f}")
        
        print("\n🚀 Next Steps:")
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
        print("   python -c \"import json; print(json.dumps(json.load(open('ml_models/evaluation/reports/training_summary.json')), indent=2))\"")
        
        print("\n📁 Project Structure Created:")
        print("- ml_models/saved/models/ - Trained ML models")
        print("- ml_models/data/ - Training datasets") 
        print("- ml_models/api/ - FastAPI application")
        print("- start_api.py - API server launcher")
        print("- test_api.py - API testing script")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = run_complete_setup()
    if success:
        print("\n✅ Setup completed successfully!")
        print("Run 'python start_api.py' to start the API server.")
    else:
        print("\n❌ Setup failed. Check the error messages above.")
        sys.exit(1)