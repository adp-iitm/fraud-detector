# Fraud & Phishing Detection ML Models

A comprehensive machine learning system for detecting phishing URLs, fraudulent emails, and suspicious transactions.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd fraud-phishing-detection

# Install dependencies
pip install -r ml_models/requirements-ml.txt

# Create directory structure
python ml_models/setup_project.py
```

### 2. Data Collection & Preprocessing

```bash
# Collect datasets
python -m ml_models.data.collectors

# Preprocess data
python -m ml_models.preprocessing.url_processor
python -m ml_models.preprocessing.email_processor  
python -m ml_models.preprocessing.transaction_processor
```

### 3. Model Training

```bash
# Train all models
python -m ml_models.training.train_url
python -m ml_models.training.train_email
python -m ml_models.training.train_transaction

# Or train everything at once
python -m ml_models.training
```

### 4. Start API Server

```bash
# Development server
python -m ml_models.api.endpoints

# Production server (with Gunicorn)
gunicorn ml_models.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📊 Model Performance

### URL Phishing Detection
- **Model**: XGBoost Classifier
- **Features**: 23 lexical and domain-based features
- **Performance**: 
  - ROC-AUC: ~0.95
  - Precision: ~0.92
  - Recall: ~0.94
  - F1-Score: ~0.93

### Email Phishing/Spam Detection  
- **Model**: Logistic Regression + TF-IDF
- **Features**: 10,000 TF-IDF features + 20 manual features
- **Performance**:
  - ROC-AUC: ~0.89
  - Precision: ~0.87
  - Recall: ~0.85
  - F1-Score: ~0.86

### Transaction Fraud Detection
- **Model**: LightGBM Classifier
- **Features**: 12 behavioral and transaction-based features
- **Performance**:
  - ROC-AUC: ~0.88
  - Precision: ~0.82
  - Recall: ~0.79
  - F1-Score: ~0.80

## 🔧 Architecture

```
ml_models/
├── config/                 # Configuration settings
├── data/                   # Data management
│   ├── collectors/         # Data collection scripts
│   ├── raw/               # Raw datasets
│   └── processed/         # Processed features
├── preprocessing/          # Feature extraction
├── models/                # Model definitions
├── training/              # Training scripts
├── inference/             # Prediction modules
├── evaluation/            # Metrics and evaluation
├── api/                   # FastAPI endpoints
├── saved/                 # Trained models
│   ├── models/            # .pkl model files
│   ├── vectorizers/       # TF-IDF vectorizers
│   └── scalers/           # Feature scalers
└── utils/                 # Utilities and helpers
```

## 🎯 Features Extracted

### URL Features
- **Lexical**: Length, special characters, digits, dots
- **Domain**: Domain length, subdomain count, TLD analysis
- **Content**: Suspicious keywords, URL shorteners
- **Security**: HTTPS usage, IP address detection

### Email Features
- **Content**: Urgent words, financial terms, personal info requests
- **Structure**: HTML tags, links, images, attachments
- **Sender**: Domain analysis, suspicious patterns
- **Linguistic**: Word length, lexical diversity, spelling errors
- **TF-IDF**: 10,000 most important n-grams (1-2 grams)

### Transaction Features
- **Amount**: Value, logarithm, Z-score vs user history
- **Temporal**: Hour, day of week, business hours
- **Merchant**: Category, risk level, online vs offline
- **Behavioral**: Transaction frequency, user age, location consistency
- **Velocity**: Recent transaction patterns, rolling statistics

## 🚀 API Usage

### URL Check
```python
import requests

response = requests.post("http://localhost:8000/api/url-check", 
                        json={"url": "https://suspicious-site.tk"})
print(response.json())
# {
#   "prediction": "phishing",
#   "confidence": 0.92,
#   "risk_score": 0.92,
#   "processing_time_ms": 45.2,
#   "risk_factors": ["Suspicious TLD", "Contains urgent words"]
# }
```

### Email Check
```python
response = requests.post("http://localhost:8000/api/email-check",
                        json={
                            "email_text": "URGENT! Verify your account now...",
                            "sender_email": "security@fake-bank.com"
                        })
print(response.json())
```

### Transaction Check
```python
response = requests.post("http://localhost:8000/api/transaction-check",
                        json={
                            "user_id": "user_123",
                            "amount": 2500.00,
                            "merchant_category": "online",
                            "merchant_name": "SuspiciousStore",
                            "location": ""
                        })
print(response.json())
```

### Batch Processing
```python
# Check multiple URLs at once
response = requests.post("http://localhost:8000/api/url-check/batch",
                        json={"urls": [
                            "https://google.com",
                            "https://phishing-site.tk",
                            "https://amazon.com"
                        ]})
```

## 🔄 Retraining Models

### Automated Retraining
```bash
# Set up automated retraining (daily)
python -m ml_models.training.retrain --schedule daily

# Manual retraining with new data
python -m ml_models.training.retrain --data-path /path/to/new/data
```

### Adding New Features
1. Modify feature extraction in `preprocessing/` modules
2. Update `config/config.py` with new feature names
3. Retrain models with new features
4. Update API schema if needed

## 📈 Monitoring & Evaluation

### Performance Monitoring
```python
# Check API performance
response = requests.get("http://localhost:8000/api/stats")
print(response.json())

# Model status
response = requests.get("http://localhost:8000/api/models/status")
```

### Model Evaluation
```bash
# Generate evaluation reports
python -m ml_models.evaluation.evaluate_models --test-data /path/to/test/data

# Compare model versions
python -m ml_models.evaluation.compare_models --model1 v1.0 --model2 v1.1
```

## 🛡️ Security Considerations

### Input Validation
- All inputs are validated and sanitized
- URL validation prevents malicious input
- Email content is processed safely
- Transaction amounts are range-checked

### Rate Limiting
- API endpoints have built-in rate limiting
- Batch processing limits prevent abuse
- Model inference times are monitored

### Privacy
- No sensitive data is logged
- Models don't store personal information
- All processing is stateless

## ⚡ Performance Optimization

### Inference Speed
- **Target**: <200ms per prediction
- **Optimizations**