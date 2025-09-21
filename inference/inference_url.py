# ml_models/inference/inference_url.py

import joblib
import numpy as np
import pandas as pd
import re
import tldextract
from urllib.parse import urlparse
from typing import Dict, List, Union, Optional
import sys
import os
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config.config import MODEL_CONFIG, MODELS_DIR, SCALERS_DIR, PREDICTION_THRESHOLDS
except ImportError:
    # Fallback configuration if config module not available
    MODEL_CONFIG = {
        'url': {
            'model_file': 'url_classifier.pkl',
            'scaler_file': 'url_scaler.pkl',
            'features': [
                'url_length', 'num_digits', 'num_special_chars', 'num_dots',
                'num_hyphens', 'num_underscores', 'num_slashes', 'num_percent',
                'has_ip', 'is_https', 'domain_length', 'tld_length',
                'num_subdomains', 'suspicious_words_count'
            ]
        }
    }
    MODELS_DIR = Path("ml_models/saved/models")
    SCALERS_DIR = Path("ml_models/saved/scalers")
    PREDICTION_THRESHOLDS = {'url': 0.5}

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        self.suspicious_words = [
            'paypal', 'ebay', 'amazon', 'microsoft', 'apple', 'google',
            'facebook', 'twitter', 'instagram', 'linkedin', 'netflix',
            'secure', 'account', 'update', 'confirm', 'verify', 'login',
            'signin', 'banking', 'credit', 'suspension', 'limited', 'urgent'
        ]
        
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.pw']
        
        self.url_shorteners = [
            'bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly',
            'short.ly', 'is.gd', 'buff.ly', 'adf.ly'
        ]
        
    def extract_features(self, url: str) -> Dict[str, Union[int, float]]:
        """Extract all features from a single URL"""
        features = {}
        
        try:
            # Clean URL
            url = url.strip().lower()
            
            # Basic URL properties
            features.update(self._extract_basic_features(url))
            
            # Domain features
            features.update(self._extract_domain_features(url))
            
            # Content-based features
            features.update(self._extract_content_features(url))
            
            # Security features
            features.update(self._extract_security_features(url))
            
            # Path and query features
            features.update(self._extract_path_features(url))
            
        except Exception as e:
            print(f"Warning: Error processing URL {url}: {e}")
            # Return default features if extraction fails
            features = self._get_default_features()
            
        return features
    
    def _extract_basic_features(self, url: str) -> Dict[str, int]:
        """Extract basic lexical features"""
        return {
            'url_length': len(url),
            'num_digits': len(re.findall(r'\d', url)),
            'num_special_chars': len(re.findall(r'[^a-zA-Z0-9]', url)),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_slashes': url.count('/'),
            'num_percent': url.count('%'),
            'num_question_marks': url.count('?'),
            'num_equals': url.count('='),
            'num_ampersands': url.count('&'),
            'num_at_symbols': url.count('@')
        }
    
    def _extract_domain_features(self, url: str) -> Dict[str, Union[int, float]]:
        """Extract domain-related features"""
        features = {}
        
        try:
            parsed = urlparse(url)
            extracted = tldextract.extract(url)
            
            # Domain properties
            domain = parsed.netloc.lower()
            features['domain_length'] = len(domain)
            features['tld_length'] = len(extracted.suffix)
            
            # Subdomain analysis
            subdomain = extracted.subdomain
            if subdomain:
                features['num_subdomains'] = len(subdomain.split('.'))
                features['subdomain_length'] = len(subdomain)
            else:
                features['num_subdomains'] = 0
                features['subdomain_length'] = 0
            
            # IP address check
            features['has_ip'] = 1 if self._is_ip_address(domain) else 0
            
            # Port usage
            features['has_port'] = 1 if ':' in domain and not domain.startswith('[') else 0
            
            # Domain reputation indicators
            features['domain_entropy'] = self._calculate_entropy(extracted.domain)
            features['subdomain_entropy'] = self._calculate_entropy(subdomain) if subdomain else 0
            
            # TLD analysis
            features['has_suspicious_tld'] = 1 if any(tld in url for tld in self.suspicious_tlds) else 0
            
        except Exception:
            features = {
                'domain_length': 0, 'tld_length': 0, 'num_subdomains': 0,
                'subdomain_length': 0, 'has_ip': 0, 'has_port': 0,
                'domain_entropy': 0, 'subdomain_entropy': 0, 'has_suspicious_tld': 0
            }
            
        return features
    
    def _extract_content_features(self, url: str) -> Dict[str, int]:
        """Extract content-based features"""
        url_lower = url.lower()
        
        # Count suspicious words
        suspicious_count = sum(1 for word in self.suspicious_words if word in url_lower)
        
        # URL shortening service detection
        is_shortener = 1 if any(service in url_lower for service in self.url_shorteners) else 0
        
        # Brand impersonation detection
        brand_words = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook']
        has_brand_impersonation = 1 if any(brand in url_lower for brand in brand_words) and not any(f'{brand}.com' in url_lower for brand in brand_words) else 0
        
        return {
            'suspicious_words_count': suspicious_count,
            'url_shortening_service': is_shortener,
            'has_brand_impersonation': has_brand_impersonation,
            'has_common_words': 1 if any(word in url_lower for word in ['login', 'secure', 'account', 'verify']) else 0
        }
    
    def _extract_security_features(self, url: str) -> Dict[str, int]:
        """Extract security-related features"""
        return {
            'is_https': 1 if url.startswith('https://') else 0,
            'has_www': 1 if 'www.' in url.lower() else 0,
            'mixed_case': 1 if any(c.isupper() for c in url) and any(c.islower() for c in url) else 0
        }
    
    def _extract_path_features(self, url: str) -> Dict[str, int]:
        """Extract path and query parameter features"""
        try:
            parsed = urlparse(url)
            path = parsed.path
            query = parsed.query
            
            return {
                'path_length': len(path),
                'num_path_segments': len([segment for segment in path.split('/') if segment]),
                'has_query_params': 1 if query else 0,
                'num_query_params': len(query.split('&')) if query else 0,
                'has_fragment': 1 if parsed.fragment else 0
            }
        except:
            return {
                'path_length': 0, 'num_path_segments': 0, 'has_query_params': 0,
                'num_query_params': 0, 'has_fragment': 0
            }
    
    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address"""
        # IPv4 pattern
        ipv4_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        
        # Remove port if present
        domain_clean = domain.split(':')[0]
        
        return bool(re.match(ipv4_pattern, domain_clean))
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0
            
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
            
        # Calculate entropy
        entropy = 0
        text_length = len(text)
        for count in char_counts.values():
            probability = count / text_length
            entropy -= probability * np.log2(probability)
            
        return entropy
    
    def _get_default_features(self) -> Dict[str, Union[int, float]]:
        """Return default feature values"""
        return {
            'url_length': 0, 'num_digits': 0, 'num_special_chars': 0,
            'num_dots': 0, 'num_hyphens': 0, 'num_underscores': 0,
            'num_slashes': 0, 'num_percent': 0, 'num_question_marks': 0,
            'num_equals': 0, 'num_ampersands': 0, 'num_at_symbols': 0,
            'domain_length': 0, 'tld_length': 0, 'num_subdomains': 0,
            'subdomain_length': 0, 'has_ip': 0, 'has_port': 0,
            'domain_entropy': 0, 'subdomain_entropy': 0, 'has_suspicious_tld': 0,
            'suspicious_words_count': 0, 'url_shortening_service': 0,
            'has_brand_impersonation': 0, 'has_common_words': 0,
            'is_https': 0, 'has_www': 0, 'mixed_case': 0,
            'path_length': 0, 'num_path_segments': 0, 'has_query_params': 0,
            'num_query_params': 0, 'has_fragment': 0
        }

class URLPredictor:
    """URL phishing detection inference"""
    
    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_extractor = URLFeatureExtractor()
        self.feature_names = MODEL_CONFIG['url']['features']
        self.threshold = PREDICTION_THRESHOLDS['url']
        self.model_path = model_path
        self.scaler_path = scaler_path
        
        # Performance tracking
        self.prediction_count = 0
        self.total_prediction_time = 0
        
    def load_model(self):
        """Load trained model and preprocessing artifacts"""
        try:
            # Determine model and scaler paths
            if self.model_path:
                model_path = Path(self.model_path)
            else:
                model_path = MODELS_DIR / MODEL_CONFIG['url']['model_file']
                
            if self.scaler_path:
                scaler_path = Path(self.scaler_path)
            else:
                scaler_path = SCALERS_DIR / MODEL_CONFIG['url']['scaler_file']
            
            # Check if files exist
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            # Load model and scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            print(f"URL model loaded successfully from {model_path}")
            
        except Exception as e:
            print(f"Error loading URL model: {e}")
            raise
    
    def predict_url(self, url: str) -> Dict[str, Union[str, float]]:
        """
        Predict if a URL is phishing or legitimate
        
        Args:
            url (str): URL to analyze
            
        Returns:
            dict: {
                "prediction": "phishing" | "legitimate",
                "confidence": float (0.0-1.0),
                "risk_score": float (0.0-1.0),
                "risk_factors": list of risk factors,
                "processing_time_ms": float
            }
        """
        start_time = time.time()
        
        # Load model if not already loaded
        if self.model is None or self.scaler is None:
            self.load_model()
        
        try:
            # Input validation
            if not url or not isinstance(url, str) or len(url.strip()) == 0:
                raise ValueError("URL cannot be empty")
            
            url = url.strip()
            
            # Extract features
            features = self.feature_extractor.extract_features(url)
            
            # Get all available features and fill missing ones with 0
            all_feature_names = list(features.keys())
            feature_vector = []
            
            # Use intersection of available features and expected features
            available_features = [fname for fname in self.feature_names if fname in features]
            
            if len(available_features) < len(self.feature_names) * 0.8:  # Less than 80% of expected features
                print(f"Warning: Only {len(available_features)}/{len(self.feature_names)} features available")
            
            # Create feature vector in the correct order
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0))
            
            # Convert to numpy array
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Handle NaN values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Get prediction probability
            prediction_proba = self.model.predict_proba(feature_vector_scaled)[0]
            
            # Extract probabilities
            prob_legitimate = prediction_proba[0]
            prob_phishing = prediction_proba[1]
            
            # Determine prediction
            is_phishing = prob_phishing > self.threshold
            
            # Calculate risk factors
            risk_factors = self._analyze_risk_factors(features, url)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Update performance tracking
            self.prediction_count += 1
            self.total_prediction_time += processing_time
            
            return {
                "prediction": "phishing" if is_phishing else "legitimate",
                "confidence": float(prob_phishing if is_phishing else prob_legitimate),
                "risk_score": float(prob_phishing),
                "risk_factors": risk_factors,
                "processing_time_ms": round(processing_time, 2),
                "features_extracted": len([f for f in features.values() if f != 0]),
                "url_length": len(url)
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            print(f"Error predicting URL {url}: {e}")
            return {
                "prediction": "legitimate",  # Safe default
                "confidence": 0.5,
                "risk_score": 0.5,
                "risk_factors": ["Error in processing"],
                "processing_time_ms": round(processing_time, 2),
                "error": str(e)
            }
    
    def _analyze_risk_factors(self, features: Dict, url: str) -> List[str]:
        """Analyze and return human-readable risk factors"""
        risk_factors = []
        
        # URL length
        if features.get('url_length', 0) > 100:
            risk_factors.append("Unusually long URL")
        
        # Suspicious TLD
        if features.get('has_suspicious_tld', 0) == 1:
            risk_factors.append("Suspicious top-level domain")
        
        # No HTTPS
        if features.get('is_https', 0) == 0:
            risk_factors.append("Not using secure HTTPS")
        
        # IP address instead of domain
        if features.get('has_ip', 0) == 1:
            risk_factors.append("Uses IP address instead of domain name")
        
        # URL shortener
        if features.get('url_shortening_service', 0) == 1:
            risk_factors.append("URL shortening service detected")
        
        # Many suspicious words
        if features.get('suspicious_words_count', 0) > 2:
            risk_factors.append("Contains multiple suspicious keywords")
        
        # Brand impersonation
        if features.get('has_brand_impersonation', 0) == 1:
            risk_factors.append("Possible brand impersonation")
        
        # Many subdomains
        if features.get('num_subdomains', 0) > 3:
            risk_factors.append("Excessive number of subdomains")
        
        # High entropy (random-looking domain)
        if features.get('domain_entropy', 0) > 4:
            risk_factors.append("Domain appears randomly generated")
        
        # Many special characters
        if features.get('num_special_chars', 0) > features.get('url_length', 1) * 0.3:
            risk_factors.append("High density of special characters")
        
        return risk_factors
    
    def predict_batch(self, urls: List[str], batch_size: int = 32) -> List[Dict]:
        """Predict multiple URLs efficiently"""
        results = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_results = []
            
            for url in batch:
                result = self.predict_url(url)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self.prediction_count == 0:
            return {"predictions": 0, "avg_time_ms": 0}
        
        return {
            "predictions": self.prediction_count,
            "avg_time_ms": self.total_prediction_time / self.prediction_count,
            "total_time_ms": self.total_prediction_time
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.prediction_count = 0
        self.total_prediction_time = 0

# Convenience function for direct use
def predict_url(url: str, model_path: Optional[str] = None, scaler_path: Optional[str] = None) -> Dict[str, Union[str, float]]:
    """Convenience function to predict a single URL"""
    predictor = URLPredictor(model_path=model_path, scaler_path=scaler_path)
    return predictor.predict_url(url)

# Example usage and testing
if __name__ == "__main__":
    print("Testing URL Inference Module...")
    print("=" * 50)
    
    # Test URLs
    test_urls = [
        "https://www.google.com",
        "https://github.com/user/repo",
        "http://paypal-security-update.tk/login.php",
        "https://amazon-account-suspended.ml/verify",
        "http://192.168.1.1:8080/admin",
        "https://bit.ly/suspicious-link",
        "http://microsoft-security-alert.ga/update"
    ]
    
    try:
        # Initialize predictor
        predictor = URLPredictor()
        
        print("Loading model...")
        predictor.load_model()
        print("Model loaded successfully!\n")
        
        # Test individual predictions
        for i, url in enumerate(test_urls, 1):
            print(f"Test {i}: {url}")
            result = predictor.predict_url(url)
            
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Risk Score: {result['risk_score']:.3f}")
            print(f"  Processing Time: {result['processing_time_ms']:.2f}ms")
            
            if result['risk_factors']:
                print(f"  Risk Factors: {', '.join(result['risk_factors'])}")
            
            print()
        
        # Test batch prediction
        print("Testing batch prediction...")
        batch_results = predictor.predict_batch(test_urls[:3])
        print(f"Processed {len(batch_results)} URLs in batch")
        
        # Performance stats
        stats = predictor.get_performance_stats()
        print(f"\nPerformance Stats:")
        print(f"  Total Predictions: {stats['predictions']}")
        print(f"  Average Time: {stats['avg_time_ms']:.2f}ms")
        
        print("\n" + "=" * 50)
        print("URL Inference testing completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("\nMake sure the model files exist:")
        print("  - ml_models/saved/models/url_classifier.pkl")
        print("  - ml_models/saved/scalers/url_scaler.pkl")
        print("\nRun the training script first if models don't exist.")