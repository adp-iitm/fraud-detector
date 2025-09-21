# ml_models/preprocessing/url_processor.py

import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import tldextract
import validators
from typing import Dict, List, Union

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        self.suspicious_words = [
            'paypal', 'ebay', 'amazon', 'microsoft', 'apple', 'google',
            'facebook', 'twitter', 'instagram', 'linkedin', 'netflix',
            'secure', 'account', 'update', 'confirm', 'verify', 'login',
            'signin', 'banking', 'credit', 'suspension', 'limited'
        ]
        
    def extract_features(self, url: str) -> Dict[str, Union[int, float]]:
        """Extract all features from a single URL"""
        features = {}
        
        try:
            # Basic URL properties
            features.update(self._extract_basic_features(url))
            
            # Domain features
            features.update(self._extract_domain_features(url))
            
            # Content-based features
            features.update(self._extract_content_features(url))
            
            # Security features
            features.update(self._extract_security_features(url))
            
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
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
            'num_ampersands': url.count('&')
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
            features['num_subdomains'] = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
            
            # IP address check
            features['has_ip'] = 1 if self._is_ip_address(domain) else 0
            
            # Port usage
            features['has_port'] = 1 if ':' in domain and not domain.startswith('[') else 0
            
            # Domain reputation indicators
            features['domain_entropy'] = self._calculate_entropy(extracted.domain)
            features['subdomain_entropy'] = self._calculate_entropy(extracted.subdomain) if extracted.subdomain else 0
            
        except Exception:
            features = {
                'domain_length': 0, 'tld_length': 0, 'num_subdomains': 0,
                'has_ip': 0, 'has_port': 0, 'domain_entropy': 0, 'subdomain_entropy': 0
            }
            
        return features
    
    def _extract_content_features(self, url: str) -> Dict[str, int]:
        """Extract content-based features"""
        url_lower = url.lower()
        
        return {
            'suspicious_words_count': sum(1 for word in self.suspicious_words if word in url_lower),
            'has_suspicious_tld': 1 if any(tld in url_lower for tld in ['.tk', '.ml', '.ga', '.cf']) else 0,
            'url_shortening_service': 1 if any(service in url_lower for service in 
                                             ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly']) else 0
        }
    
    def _extract_security_features(self, url: str) -> Dict[str, int]:
        """Extract security-related features"""
        return {
            'is_https': 1 if url.startswith('https://') else 0,
            'has_www': 1 if 'www.' in url.lower() else 0
        }
    
    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address"""
        ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        return bool(re.match(ip_pattern, domain))
    
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
            'num_equals': 0, 'num_ampersands': 0, 'domain_length': 0,
            'tld_length': 0, 'num_subdomains': 0, 'has_ip': 0,
            'has_port': 0, 'domain_entropy': 0, 'subdomain_entropy': 0,
            'suspicious_words_count': 0, 'has_suspicious_tld': 0,
            'url_shortening_service': 0, 'is_https': 0, 'has_www': 0
        }
    
    def process_batch(self, urls: List[str]) -> pd.DataFrame:
        """Process multiple URLs and return DataFrame"""
        features_list = []
        for url in urls:
            features = self.extract_features(url)
            features['url'] = url
            features_list.append(features)
            
        return pd.DataFrame(features_list)

def preprocess_url_dataset(input_file: str, output_file: str) -> None:
    """Preprocess URL dataset and save features"""
    extractor = URLFeatureExtractor()
    
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Extract features
    print("Extracting URL features...")
    feature_df = extractor.process_batch(df['url'].tolist())
    
    # Merge with labels
    result_df = pd.merge(feature_df, df[['url', 'label']], on='url')
    
    # Save processed data
    result_df.to_csv(output_file, index=False)
    print(f"Processed URL dataset saved to {output_file}")
    print(f"Dataset shape: {result_df.shape}")

if __name__ == "__main__":
    # Example usage
    extractor = URLFeatureExtractor()
    
    test_urls = [
        "https://www.google.com",
        "http://bit.ly/suspicious-link",
        "https://paypal-security-update.tk/login",
        "http://192.168.1.1:8080/admin"
    ]
    
    for url in test_urls:
        features = extractor.extract_features(url)
        print(f"\nURL: {url}")
        print(f"Features: {features}")