from fastapi import FastAPI, HTTPException
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
    features['num_digits'] = len(re.findall(r'\\d', url))
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
