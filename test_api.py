#!/usr/bin/env python3
import requests
import json
import time
from datetime import datetime

def test_api():
    base_url = "http://localhost:8000"
    
    print("Testing Fraud & Phishing Detection API")
    print("=" * 50)
    
    # Test health
    print("\n1. Testing Health endpoint...")
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
    print("\n2. Testing URL check...")
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
    print("\n3. Testing Email check...")
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
    print("\n4. Testing Transaction check...")
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
    print("\n5. Testing Batch URL check...")
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
    
    print("\n" + "=" * 50)
    print("API testing complete!")

if __name__ == "__main__":
    print("Starting tests in 3 seconds...")
    time.sleep(3)
    test_api()
