# ml_models/api/endpoints.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Union, Optional
import asyncio
import time
from datetime import datetime
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.inference_url import URLPredictor
from inference.inference_email import EmailPredictor
from inference.inference_transaction import TransactionPredictor
from config.config import PERFORMANCE_CONFIG

# Initialize FastAPI app
app = FastAPI(
    title="Fraud & Phishing Detection API",
    description="Machine Learning API for detecting phishing URLs, fraudulent emails, and suspicious transactions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictors (loaded on startup)
url_predictor = None
email_predictor = None
transaction_predictor = None

# Request/Response Models
class URLCheckRequest(BaseModel):
    url: str = Field(..., description="URL to check for phishing")
    
    @validator('url')
    def validate_url(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("URL cannot be empty")
        return v.strip()

class EmailCheckRequest(BaseModel):
    email_text: str = Field(..., description="Email content to analyze")
    sender_email: Optional[str] = Field(None, description="Sender email address")
    subject: Optional[str] = Field(None, description="Email subject")
    
    @validator('email_text')
    def validate_email_text(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Email text cannot be empty")
        return v.strip()

class TransactionCheckRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp (ISO format)")
    merchant_category: str = Field(..., description="Merchant category")
    merchant_name: str = Field(..., description="Merchant name")
    location: Optional[str] = Field("", description="Transaction location")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Invalid timestamp format. Use ISO format.")
        return v

class BatchURLCheckRequest(BaseModel):
    urls: List[str] = Field(..., max_items=100, description="List of URLs to check")

class BatchEmailCheckRequest(BaseModel):
    emails: List[EmailCheckRequest] = Field(..., max_items=50, description="List of emails to check")

class BatchTransactionCheckRequest(BaseModel):
    transactions: List[TransactionCheckRequest] = Field(..., max_items=100, description="List of transactions to check")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Prediction result")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(default="1.0.0", description="Model version used")

class URLPredictionResponse(PredictionResponse):
    risk_factors: Optional[List[str]] = Field(default=[], description="Identified risk factors")
    features_extracted: Optional[int] = Field(default=0, description="Number of features extracted")

class EmailPredictionResponse(PredictionResponse):
    risk_factors: Optional[List[str]] = Field(default=[], description="Identified risk factors")
    manual_features_count: Optional[int] = Field(default=0, description="Number of manual features")

class TransactionPredictionResponse(PredictionResponse):
    risk_factors: Optional[List[str]] = Field(default=[], description="Identified risk factors")
    amount_zscore: Optional[float] = Field(default=0.0, description="Amount Z-score")

class BatchPredictionResponse(BaseModel):
    results: List[Union[URLPredictionResponse, EmailPredictionResponse, TransactionPredictionResponse]]
    total_processed: int
    total_processing_time_ms: float
    average_processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    version: str

# Startup event to load models
@app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    global url_predictor, email_predictor, transaction_predictor
    
    print("Loading ML models...")
    
    try:
        # Load predictors
        url_predictor = URLPredictor()
        url_predictor.load_model()
        print("✓ URL model loaded")
    except Exception as e:
        print(f"✗ Failed to load URL model: {e}")
        url_predictor = None
    
    try:
        email_predictor = EmailPredictor()
        email_predictor.load_model()
        print("✓ Email model loaded")
    except Exception as e:
        print(f"✗ Failed to load Email model: {e}")
        email_predictor = None
    
    try:
        transaction_predictor = TransactionPredictor()
        transaction_predictor.load_model()
        print("✓ Transaction model loaded")
    except Exception as e:
        print(f"✗ Failed to load Transaction model: {e}")
        transaction_predictor = None
    
    print("Startup complete!")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={
            "url": url_predictor is not None,
            "email": email_predictor is not None,
            "transaction": transaction_predictor is not None
        },
        version="1.0.0"
    )

# URL Check Endpoints
@app.post("/api/url-check", response_model=URLPredictionResponse)
async def check_url(request: URLCheckRequest):
    """Check if a URL is phishing or legitimate"""
    if url_predictor is None:
        raise HTTPException(status_code=503, detail="URL model not available")
    
    start_time = time.time()
    
    try:
        # Predict URL
        result = url_predictor.predict_url(request.url)
        processing_time = (time.time() - start_time) * 1000
        
        # Check if processing time exceeds threshold
        if processing_time > PERFORMANCE_CONFIG['max_inference_time_ms']:
            print(f"Warning: URL prediction took {processing_time:.2f}ms (threshold: {PERFORMANCE_CONFIG['max_inference_time_ms']}ms)")
        
        return URLPredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            risk_score=result['risk_score'],
            processing_time_ms=processing_time,
            risk_factors=result.get('risk_factors', []),
            features_extracted=result.get('features_extracted', 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL prediction failed: {str(e)}")

@app.post("/api/url-check/batch", response_model=BatchPredictionResponse)
async def check_urls_batch(request: BatchURLCheckRequest):
    """Check multiple URLs in batch"""
    if url_predictor is None:
        raise HTTPException(status_code=503, detail="URL model not available")
    
    start_time = time.time()
    results = []
    
    try:
        for url in request.urls:
            url_start_time = time.time()
            result = url_predictor.predict_url(url)
            url_processing_time = (time.time() - url_start_time) * 1000
            
            results.append(URLPredictionResponse(
                prediction=result['prediction'],
                confidence=result['confidence'],
                risk_score=result['risk_score'],
                processing_time_ms=url_processing_time,
                risk_factors=result.get('risk_factors', []),
                features_extracted=result.get('features_extracted', 0)
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            results=results,
            total_processed=len(results),
            total_processing_time_ms=total_time,
            average_processing_time_ms=total_time / len(results) if results else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch URL prediction failed: {str(e)}")

# Email Check Endpoints  
@app.post("/api/email-check", response_model=EmailPredictionResponse)
async def check_email(request: EmailCheckRequest):
    """Check if an email is phishing/spam or legitimate"""
    if email_predictor is None:
        raise HTTPException(status_code=503, detail="Email model not available")
    
    start_time = time.time()
    
    try:
        # Predict email
        result = email_predictor.predict_email(request.email_text, request.sender_email or "")
        processing_time = (time.time() - start_time) * 1000
        
        return EmailPredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            risk_score=result['risk_score'],
            processing_time_ms=processing_time,
            risk_factors=result.get('risk_factors', []),
            manual_features_count=result.get('manual_features_count', 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email prediction failed: {str(e)}")

@app.post("/api/email-check/batch", response_model=BatchPredictionResponse)
async def check_emails_batch(request: BatchEmailCheckRequest):
    """Check multiple emails in batch"""
    if email_predictor is None:
        raise HTTPException(status_code=503, detail="Email model not available")
    
    start_time = time.time()
    results = []
    
    try:
        for email_req in request.emails:
            email_start_time = time.time()
            result = email_predictor.predict_email(email_req.email_text, email_req.sender_email or "")
            email_processing_time = (time.time() - email_start_time) * 1000
            
            results.append(EmailPredictionResponse(
                prediction=result['prediction'],
                confidence=result['confidence'],
                risk_score=result['risk_score'],
                processing_time_ms=email_processing_time,
                risk_factors=result.get('risk_factors', []),
                manual_features_count=result.get('manual_features_count', 0)
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            results=results,
            total_processed=len(results),
            total_processing_time_ms=total_time,
            average_processing_time_ms=total_time / len(results) if results else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch email prediction failed: {str(e)}")

# Transaction Check Endpoints
@app.post("/api/transaction-check", response_model=TransactionPredictionResponse)
async def check_transaction(request: TransactionCheckRequest):
    """Check if a transaction is fraudulent or legitimate"""
    if transaction_predictor is None:
        raise HTTPException(status_code=503, detail="Transaction model not available")
    
    start_time = time.time()
    
    try:
        # Convert request to dictionary
        transaction_data = {
            'user_id': request.user_id,
            'amount': request.amount,
            'timestamp': request.timestamp or datetime.now().isoformat(),
            'merchant_category': request.merchant_category,
            'merchant_name': request.merchant_name,
            'location': request.location
        }
        
        # Predict transaction (without user history for now)
        result = transaction_predictor.predict_transaction(transaction_data)
        processing_time = (time.time() - start_time) * 1000
        
        return TransactionPredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            risk_score=result['risk_score'],
            processing_time_ms=processing_time,
            risk_factors=result.get('risk_factors', []),
            amount_zscore=result.get('amount_zscore', 0.0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transaction prediction failed: {str(e)}")

@app.post("/api/transaction-check/batch", response_model=BatchPredictionResponse)
async def check_transactions_batch(request: BatchTransactionCheckRequest):
    """Check multiple transactions in batch"""
    if transaction_predictor is None:
        raise HTTPException(status_code=503, detail="Transaction model not available")
    
    start_time = time.time()
    results = []
    
    try:
        for txn_req in request.transactions:
            txn_start_time = time.time()
            
            transaction_data = {
                'user_id': txn_req.user_id,
                'amount': txn_req.amount,
                'timestamp': txn_req.timestamp or datetime.now().isoformat(),
                'merchant_category': txn_req.merchant_category,
                'merchant_name': txn_req.merchant_name,
                'location': txn_req.location
            }
            
            result = transaction_predictor.predict_transaction(transaction_data)
            txn_processing_time = (time.time() - txn_start_time) * 1000
            
            results.append(TransactionPredictionResponse(
                prediction=result['prediction'],
                confidence=result['confidence'],
                risk_score=result['risk_score'],
                processing_time_ms=txn_processing_time,
                risk_factors=result.get('risk_factors', []),
                amount_zscore=result.get('amount_zscore', 0.0)
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            results=results,
            total_processed=len(results),
            total_processing_time_ms=total_time,
            average_processing_time_ms=total_time / len(results) if results else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch transaction prediction failed: {str(e)}")

# Model Management Endpoints
@app.post("/api/models/reload")
async def reload_models():
    """Reload all ML models"""
    global url_predictor, email_predictor, transaction_predictor
    
    results = {
        'url': False,
        'email': False,
        'transaction': False,
        'timestamp': datetime.now().isoformat()
    }
    
    # Reload URL model
    try:
        url_predictor = URLPredictor()
        url_predictor.load_model()
        results['url'] = True
    except Exception as e:
        print(f"Failed to reload URL model: {e}")
    
    # Reload Email model
    try:
        email_predictor = EmailPredictor()
        email_predictor.load_model()
        results['email'] = True
    except Exception as e:
        print(f"Failed to reload Email model: {e}")
    
    # Reload Transaction model
    try:
        transaction_predictor = TransactionPredictor()
        transaction_predictor.load_model()
        results['transaction'] = True
    except Exception as e:
        print(f"Failed to reload Transaction model: {e}")
    
    return {"status": "completed", "results": results}

@app.get("/api/models/status")
async def get_models_status():
    """Get current model status and performance metrics"""
    return {
        "models": {
            "url": {
                "loaded": url_predictor is not None,
                "model_type": "XGBoost" if url_predictor else None
            },
            "email": {
                "loaded": email_predictor is not None,
                "model_type": "Logistic Regression" if email_predictor else None
            },
            "transaction": {
                "loaded": transaction_predictor is not None,
                "model_type": "LightGBM" if transaction_predictor else None
            }
        },
        "performance_config": PERFORMANCE_CONFIG,
        "timestamp": datetime.now().isoformat()
    }

# Statistics and Monitoring Endpoints
@app.get("/api/stats")
async def get_api_stats():
    """Get API usage statistics"""
    # In a real implementation, you'd track these metrics
    return {
        "total_predictions": 0,
        "predictions_by_type": {
            "url": 0,
            "email": 0,
            "transaction": 0
        },
        "average_response_times": {
            "url": 0.0,
            "email": 0.0,
            "transaction": 0.0
        },
        "error_rate": 0.0,
        "uptime": "N/A",
        "last_updated": datetime.now().isoformat()
    }

# Background task for model warming
async def warm_up_models():
    """Warm up models by running test predictions"""
    print("Warming up models...")
    
    # Warm up URL predictor
    if url_predictor:
        try:
            url_predictor.predict_url("https://example.com")
            print("✓ URL model warmed up")
        except Exception as e:
            print(f"✗ URL model warm-up failed: {e}")
    
    # Warm up Email predictor
    if email_predictor:
        try:
            email_predictor.predict_email("Test email content", "test@example.com")
            print("✓ Email model warmed up")
        except Exception as e:
            print(f"✗ Email model warm-up failed: {e}")
    
    # Warm up Transaction predictor
    if transaction_predictor:
        try:
            test_transaction = {
                'user_id': 'test_user',
                'amount': 100.0,
                'timestamp': datetime.now().isoformat(),
                'merchant_category': 'grocery',
                'merchant_name': 'TestMart',
                'location': 'Test City'
            }
            transaction_predictor.predict_transaction(test_transaction)
            print("✓ Transaction model warmed up")
        except Exception as e:
            print(f"✗ Transaction model warm-up failed: {e}")

@app.post("/api/models/warmup")
async def warmup_models(background_tasks: BackgroundTasks):
    """Trigger model warm-up in background"""
    background_tasks.add_task(warm_up_models)
    return {"status": "warmup initiated", "timestamp": datetime.now().isoformat()}

# Error handlers
@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    return {
        "error": "Validation Error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors"""
    return {
        "error": "Internal Server Error",
        "detail": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }

# Main entry point for testing
if __name__ == "__main__":
    import uvicorn
    
    print("Starting Fraud & Phishing Detection API...")
    print("API Documentation will be available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Alternative startup script for production
# ml_models/api/main.py

"""
Production startup script for the Fraud & Phishing Detection API

Usage:
    python -m ml_models.api.main
    
Or with gunicorn:
    gunicorn ml_models.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_models.api.endpoints import app

# Configuration for production
if os.getenv("ENVIRONMENT") == "production":
    # Production configurations
    app.debug = False
    
    # Update CORS for production
    from fastapi.middleware.cors import CORSMiddleware
    
    # Remove permissive CORS and add specific origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://yourdomain.com"],  # Specify your frontend domain
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

if __name__ == "__main__":
    import uvicorn
    
    # Development server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )