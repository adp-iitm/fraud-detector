# ml_models/preprocessing/email_processor.py

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import email
from email.mime.text import MIMEText
import joblib

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class EmailFeatureExtractor:
    """Extract features from emails for phishing/spam detection"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Suspicious patterns
        self.urgent_words = [
            'urgent', 'immediate', 'asap', 'expire', 'expires', 'suspended',
            'limited time', 'act now', 'hurry', 'rush', 'deadline'
        ]
        
        self.financial_words = [
            'money', 'cash', 'credit', 'bank', 'account', 'payment', 'transfer',
            'loan', 'invest', 'profit', 'refund', 'claim', 'winner', 'prize'
        ]
        
        self.suspicious_domains = [
            'tempmail', '10minutemail', 'guerrillamail', 'mailinator',
            'throwaway', 'temp', 'disposable'
        ]
        
    def extract_features(self, email_content: str, sender_email: str = "") -> Dict[str, Union[int, float]]:
        """Extract all features from email content"""
        features = {}
        
        try:
            # Clean and parse email
            clean_text = self._clean_email_text(email_content)
            
            # Basic features
            features.update(self._extract_basic_features(email_content, clean_text))
            
            # Content features
            features.update(self._extract_content_features(clean_text))
            
            # Sender features
            features.update(self._extract_sender_features(sender_email))
            
            # HTML features
            features.update(self._extract_html_features(email_content))
            
            # Linguistic features
            features.update(self._extract_linguistic_features(clean_text))
            
        except Exception as e:
            print(f"Error processing email: {e}")
            features = self._get_default_features()
            
        return features
    
    def _clean_email_text(self, email_content: str) -> str:
        """Clean and preprocess email text"""
        # Remove HTML tags
        if '<html>' in email_content.lower() or '<body>' in email_content.lower():
            soup = BeautifulSoup(email_content, 'html.parser')
            text = soup.get_text()
        else:
            text = email_content
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove email headers if present
        text = re.sub(r'^(From:|To:|Subject:|Date:).*?$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _extract_basic_features(self, raw_content: str, clean_text: str) -> Dict[str, Union[int, float]]:
        """Extract basic email features"""
        return {
            'email_length': len(clean_text),
            'num_words': len(clean_text.split()),
            'num_sentences': len(re.findall(r'[.!?]+', clean_text)),
            'num_paragraphs': len(re.findall(r'\n\s*\n', raw_content)),
            'num_links': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', raw_content)),
            'num_images': len(re.findall(r'<img', raw_content, re.IGNORECASE)),
            'num_attachments': len(re.findall(r'attachment|attached', clean_text, re.IGNORECASE))
        }
    
    def _extract_content_features(self, text: str) -> Dict[str, int]:
        """Extract content-based features"""
        text_lower = text.lower()
        
        return {
            'num_urgent_words': sum(1 for word in self.urgent_words if word in text_lower),
            'num_financial_words': sum(1 for word in self.financial_words if word in text_lower),
            'has_money_symbol': 1 if any(symbol in text for symbol in ['$', '€', '£', '¥']) else 0,
            'num_exclamation': text.count('!'),
            'num_caps_words': len([word for word in text.split() if word.isupper() and len(word) > 2]),
            'has_personal_info_request': 1 if any(phrase in text_lower for phrase in 
                                                ['social security', 'ssn', 'credit card', 'password', 'pin']) else 0
        }
    
    def _extract_sender_features(self, sender_email: str) -> Dict[str, int]:
        """Extract sender-based features"""
        if not sender_email:
            return {
                'sender_suspicious_domain': 0,
                'sender_has_numbers': 0,
                'sender_length': 0
            }
        
        domain = sender_email.split('@')[-1] if '@' in sender_email else ""
        
        return {
            'sender_suspicious_domain': 1 if any(susp in domain.lower() for susp in self.suspicious_domains) else 0,
            'sender_has_numbers': 1 if any(char.isdigit() for char in sender_email) else 0,
            'sender_length': len(sender_email)
        }
    
    def _extract_html_features(self, email_content: str) -> Dict[str, int]:
        """Extract HTML-specific features"""
        return {
            'is_html': 1 if any(tag in email_content.lower() for tag in ['<html>', '<body>', '<div>']) else 0,
            'num_html_tags': len(re.findall(r'<[^>]+>', email_content)),
            'has_javascript': 1 if '<script' in email_content.lower() else 0,
            'has_forms': 1 if '<form' in email_content.lower() else 0,
            'has_hidden_text': 1 if 'display:none' in email_content.lower() or 'visibility:hidden' in email_content.lower() else 0
        }
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic and stylistic features"""
        words = text.split()
        if not words:
            return {'avg_word_length': 0, 'lexical_diversity': 0, 'spelling_errors': 0}
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Lexical diversity (unique words / total words)
        unique_words = set(word.lower() for word in words)
        lexical_diversity = len(unique_words) / len(words)
        
        # Simple spelling error detection (words with numbers mixed in)
        spelling_errors = len([word for word in words if re.search(r'[a-zA-Z][0-9]|[0-9][a-zA-Z]', word)])
        
        return {
            'avg_word_length': avg_word_length,
            'lexical_diversity': lexical_diversity,
            'spelling_errors': spelling_errors
        }
    
    def _get_default_features(self) -> Dict[str, Union[int, float]]:
        """Return default feature values"""
        return {
            'email_length': 0, 'num_words': 0, 'num_sentences': 0, 'num_paragraphs': 0,
            'num_links': 0, 'num_images': 0, 'num_attachments': 0, 'num_urgent_words': 0,
            'num_financial_words': 0, 'has_money_symbol': 0, 'num_exclamation': 0,
            'num_caps_words': 0, 'has_personal_info_request': 0, 'sender_suspicious_domain': 0,
            'sender_has_numbers': 0, 'sender_length': 0, 'is_html': 0, 'num_html_tags': 0,
            'has_javascript': 0, 'has_forms': 0, 'has_hidden_text': 0, 'avg_word_length': 0,
            'lexical_diversity': 0, 'spelling_errors': 0
        }

class EmailTextProcessor:
    """Process email text for TF-IDF vectorization"""
    
    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self.vectorizer = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for vectorization"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def fit_vectorizer(self, texts: List[str]) -> None:
        """Fit TF-IDF vectorizer on training texts"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        self.vectorizer.fit(cleaned_texts)
    
    def transform_texts(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors"""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned_texts).toarray()
    
    def save_vectorizer(self, filepath: str) -> None:
        """Save fitted vectorizer"""
        joblib.dump(self.vectorizer, filepath)
    
    def load_vectorizer(self, filepath: str) -> None:
        """Load fitted vectorizer"""
        self.vectorizer = joblib.load(filepath)

def preprocess_email_dataset(input_file: str, output_file: str, vectorizer_file: str) -> None:
    """Preprocess email dataset and save features"""
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Initialize processors
    feature_extractor = EmailFeatureExtractor()
    text_processor = EmailTextProcessor()
    
    print("Extracting email features...")
    
    # Extract manual features
    features_list = []
    for idx, row in df.iterrows():
        email_text = row.get('email_text', '')
        sender = row.get('sender', '')
        
        features = feature_extractor.extract_features(email_text, sender)
        features['email_id'] = idx
        features_list.append(features)
    
    # Create features DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Fit and transform TF-IDF
    print("Processing TF-IDF features...")
    text_processor.fit_vectorizer(df['email_text'].tolist())
    tfidf_features = text_processor.transform_texts(df['email_text'].tolist())
    
    # Create TF-IDF DataFrame
    tfidf_columns = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_columns)
    tfidf_df['email_id'] = range(len(tfidf_df))
    
    # Combine features
    combined_df = pd.merge(features_df, tfidf_df, on='email_id')
    combined_df = pd.merge(combined_df, df[['label']].reset_index().rename(columns={'index': 'email_id'}), on='email_id')
    
    # Save processed data and vectorizer
    combined_df.to_csv(output_file, index=False)
    text_processor.save_vectorizer(vectorizer_file)
    
    print(f"Processed email dataset saved to {output_file}")
    print(f"TF-IDF vectorizer saved to {vectorizer_file}")
    print(f"Dataset shape: {combined_df.shape}")

if __name__ == "__main__":
    # Example usage
    extractor = EmailFeatureExtractor()
    
    test_emails = [
        {
            'content': "URGENT! Your account will be suspended. Click here to verify: http://fake-bank.com",
            'sender': "security@temp-mail.com"
        },
        {
            'content': "Hi there, thanks for your order. Your package will arrive tomorrow.",
            'sender': "support@company.com"
        }
    ]
    
    for i, email_data in enumerate(test_emails):
        features = extractor.extract_features(email_data['content'], email_data['sender'])
        print(f"\nEmail {i+1}: {email_data['content'][:50]}...")
        print(f"Features: {features}")