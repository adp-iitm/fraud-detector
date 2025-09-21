# ml_models/data/collectors/url_collector.py

import requests
import pandas as pd
import time
from typing import List, Tuple
import os
from pathlib import Path

class URLDataCollector:
    """Collect phishing and legitimate URLs from various sources"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_phishtank_data(self) -> pd.DataFrame:
        """Collect data from PhishTank"""
        print("Collecting PhishTank data...")
        
        try:
            # PhishTank requires registration for API access
            # For demo purposes, we'll create sample data
            phishing_urls = [
                "http://paypal-security.tk/login.php",
                "https://amazon-account-suspended.ml/verify",
                "http://apple-id-locked.cf/unlock.html",
                "https://microsoft-security-alert.ga/update",
                "http://facebook-security-check.tk/login"
            ]
            
            df = pd.DataFrame({
                'url': phishing_urls,
                'label': ['phishing'] * len(phishing_urls),
                'source': ['phishtank'] * len(phishing_urls)
            })
            
            return df
            
        except Exception as e:
            print(f"Error collecting PhishTank data: {e}")
            return pd.DataFrame(columns=['url', 'label', 'source'])
    
    def collect_legitimate_urls(self) -> pd.DataFrame:
        """Collect legitimate URLs from popular websites"""
        print("Collecting legitimate URLs...")
        
        legitimate_urls = [
            "https://www.google.com",
            "https://www.facebook.com",
            "https://www.twitter.com",
            "https://www.linkedin.com",
            "https://www.github.com",
            "https://www.stackoverflow.com",
            "https://www.wikipedia.org",
            "https://www.amazon.com",
            "https://www.microsoft.com",
            "https://www.apple.com",
            "https://www.netflix.com",
            "https://www.youtube.com",
            "https://www.reddit.com",
            "https://www.instagram.com",
            "https://www.paypal.com",
            "https://www.ebay.com",
            "https://www.cnn.com",
            "https://www.bbc.com",
            "https://www.nytimes.com",
            "https://www.adobe.com"
        ]
        
        df = pd.DataFrame({
            'url': legitimate_urls,
            'label': ['legitimate'] * len(legitimate_urls),
            'source': ['manual'] * len(legitimate_urls)
        })
        
        return df
    
    def collect_openphish_data(self) -> pd.DataFrame:
        """Collect data from OpenPhish (real feed if available, else fallback)"""
        print("Collecting OpenPhish data...")
        try:
            feed_url = "https://openphish.com/feed.txt"
            resp = requests.get(feed_url, timeout=20)
            if resp.status_code == 200 and resp.text:
                urls = [u.strip() for u in resp.text.splitlines() if u.strip()]
                df = pd.DataFrame({
                    'url': urls,
                    'label': ['phishing'] * len(urls),
                    'source': ['openphish'] * len(urls)
                })
                return df
            else:
                print(f"OpenPhish request returned status {resp.status_code}, using fallback sample data")
        except Exception as e:
            print(f"Error fetching OpenPhish feed ({e}), using fallback sample data")
        
        # Fallback sample data
        openphish_urls = [
            "http://secure-login-update.tk/paypal/",
            "https://account-verification.ml/amazon/signin",
            "http://security-alert.cf/microsoft/update",
            "https://suspended-account.ga/ebay/restore",
            "http://urgent-action-required.tk/apple/verify"
        ]
        df = pd.DataFrame({
            'url': openphish_urls,
            'label': ['phishing'] * len(openphish_urls),
            'source': ['openphish'] * len(openphish_urls)
        })
        return df
    
    def collect_all_url_data(self) -> pd.DataFrame:
        """Collect data from all sources"""
        all_data = []
        
        # Collect from different sources
        phishtank_data = self.collect_phishtank_data()
        if not phishtank_data.empty:
            all_data.append(phishtank_data)
            
        openphish_data = self.collect_openphish_data()
        if not openphish_data.empty:
            all_data.append(openphish_data)
            
        legitimate_data = self.collect_legitimate_urls()
        if not legitimate_data.empty:
            all_data.append(legitimate_data)
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save raw data
            output_file = self.data_dir / "raw_urls.csv"
            combined_df.to_csv(output_file, index=False)
            print(f"Saved {len(combined_df)} URLs to {output_file}")
            
            return combined_df
        else:
            return pd.DataFrame(columns=['url', 'label', 'source'])

# ml_models/data/collectors/email_collector.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

class EmailDataCollector:
    """Collect phishing/spam and legitimate emails"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_phishing_emails(self) -> List[Dict]:
        """Generate sample phishing emails"""
        phishing_emails = [
            {
                'email_text': """URGENT: Your PayPal account has been suspended due to suspicious activity. 
                Click here immediately to verify your account: http://paypal-verify.tk/login
                Failure to act within 24 hours will result in permanent account closure.
                Account Security Team""",
                'sender': 'security@paypal-alerts.com',
                'subject': 'URGENT: Account Suspended - Immediate Action Required',
                'label': 'phishing'
            },
            {
                'email_text': """Congratulations! You have won $1,000,000 in the Microsoft Lottery!
                To claim your prize, please provide your banking details and send $500 processing fee.
                Contact: winner@microsoft-lottery.tk""",
                'sender': 'lottery@microsoft-winner.com',
                'subject': 'YOU WON $1,000,000!!!',
                'label': 'phishing'
            },
            {
                'email_text': """Your Amazon account will be closed tomorrow unless you update your payment information.
                Click here: http://amazon-account-update.ml/payment
                Amazon Customer Service""",
                'sender': 'noreply@amazon-security.net',
                'subject': 'Account Closure Notice - Update Required',
                'label': 'phishing'
            },
            {
                'email_text': """FINAL NOTICE: Your bank account has been compromised. 
                Verify your identity immediately: http://secure-bank-verify.cf/login
                Enter your username, password, and SSN to prevent account closure.""",
                'sender': 'security@bank-alerts.org',
                'subject': 'FINAL NOTICE: Account Compromised',
                'label': 'phishing'
            },
            {
                'email_text': """You have received a secure document. Click to download:
                http://document-viewer.tk/download?id=12345
                This link expires in 2 hours.""",
                'sender': 'documents@secure-share.net',
                'subject': 'Secure Document Delivery',
                'label': 'phishing'
            }
        ]
        return phishing_emails
    
    def generate_legitimate_emails(self) -> List[Dict]:
        """Generate sample legitimate emails"""
        legitimate_emails = [
            {
                'email_text': """Thank you for your recent order. Your package has been shipped and will arrive within 3-5 business days.
                Track your package: [legitimate tracking link]
                Order #12345
                Customer Service Team""",
                'sender': 'orders@company.com',
                'subject': 'Your Order Has Shipped',
                'label': 'legitimate'
            },
            {
                'email_text': """Your monthly statement is now available online.
                Login to your account to view your statement.
                If you have questions, please contact us at 1-800-123-4567.
                Best regards, Customer Service""",
                'sender': 'statements@bank.com',
                'subject': 'Monthly Statement Available',
                'label': 'legitimate'
            },
            {
                'email_text': """Welcome to our newsletter! Here are this week's top articles:
                1. 10 Tips for Better Productivity
                2. New Product Launch Announcement
                3. Customer Success Story
                Unsubscribe anytime by clicking here.""",
                'sender': 'newsletter@company.com',
                'subject': 'Weekly Newsletter - Week of Jan 15',
                'label': 'legitimate'
            },
            {
                'email_text': """Your appointment has been confirmed for January 20th at 2:00 PM.
                Location: 123 Main Street, Suite 100
                Please arrive 15 minutes early and bring a valid ID.
                To reschedule, please call 555-0123.""",
                'sender': 'appointments@clinic.com',
                'subject': 'Appointment Confirmation',
                'label': 'legitimate'
            },
            {
                'email_text': """Your password has been successfully changed.
                If you did not make this change, please contact support immediately.
                For security purposes, please do not reply to this email.
                Account Security Team""",
                'sender': 'noreply@company.com',
                'subject': 'Password Changed Successfully',
                'label': 'legitimate'
            }
        ]
        return legitimate_emails
    
    def collect_all_email_data(self) -> pd.DataFrame:
        """Collect all email data"""
        print("Generating email dataset...")
        
        # Collect emails
        phishing_emails = self.generate_phishing_emails()
        legitimate_emails = self.generate_legitimate_emails()
        
        # Combine and create DataFrame
        all_emails = phishing_emails + legitimate_emails
        df = pd.DataFrame(all_emails)
        
        # Add additional samples by modifying existing ones
        additional_emails = []
        for _ in range(10):  # Create 10 additional variations
            base_email = np.random.choice(all_emails)
            modified_email = base_email.copy()
            # Simple modifications for variety
            modified_email['email_text'] = base_email['email_text'].replace('Click', 'Please click')
            additional_emails.append(modified_email)
        
        additional_df = pd.DataFrame(additional_emails)
        final_df = pd.concat([df, additional_df], ignore_index=True)
        
        # Save raw data
        output_file = self.data_dir / "raw_emails.csv"
        final_df.to_csv(output_file, index=False)
        print(f"Saved {len(final_df)} emails to {output_file}")
        
        return final_df

# ml_models/data/collectors/transaction_collector.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random

class TransactionDataCollector:
    """Generate synthetic transaction data for fraud detection"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.merchant_categories = [
            'grocery', 'restaurant', 'gas_station', 'retail', 'online',
            'entertainment', 'travel', 'healthcare', 'utilities', 'atm'
        ]
        
        self.merchants = {
            'grocery': ['SuperMart', 'FreshFoods', 'GreenGrocer', 'QuickStop'],
            'restaurant': ['PizzaPalace', 'BurgerKing', 'TacoBell', 'CafeCorner'],
            'gas_station': ['Shell', 'Exxon', 'BP', 'Chevron'],
            'retail': ['WalMart', 'Target', 'BestBuy', 'HomeDepot'],
            'online': ['Amazon', 'eBay', 'Etsy', 'OnlineStore'],
            'entertainment': ['MovieTheater', 'GameStore', 'MusicShop', 'BookStore'],
            'travel': ['AirlineTicket', 'Hotel', 'CarRental', 'TravelAgency'],
            'healthcare': ['Pharmacy', 'Clinic', 'Hospital', 'DentalOffice'],
            'utilities': ['ElectricCo', 'WaterDept', 'PhoneCompany', 'InternetISP'],
            'atm': ['BankATM', 'ConvenienceATM', 'MallATM', 'AirportATM']
        }
        
    def generate_legitimate_transactions(self, num_users: int = 100, transactions_per_user: int = 50) -> pd.DataFrame:
        """Generate legitimate transaction patterns"""
        transactions = []
        
        for user_id in range(num_users):
            user_transactions = self._generate_user_transactions(
                f"user_{user_id}", 
                transactions_per_user, 
                is_fraudulent=False
            )
            transactions.extend(user_transactions)
            
        return pd.DataFrame(transactions)
    
    def generate_fraudulent_transactions(self, num_users: int = 20, transactions_per_user: int = 10) -> pd.DataFrame:
        """Generate fraudulent transaction patterns"""
        transactions = []
        
        for user_id in range(num_users):
            user_transactions = self._generate_user_transactions(
                f"fraud_user_{user_id}", 
                transactions_per_user, 
                is_fraudulent=True
            )
            transactions.extend(user_transactions)
            
        return pd.DataFrame(transactions)
    
    def _generate_user_transactions(self, user_id: str, num_transactions: int, is_fraudulent: bool = False) -> list:
        """Generate transactions for a single user"""
        transactions = []
        base_time = datetime.now() - timedelta(days=30)
        
        # User behavioral patterns
        if is_fraudulent:
            # Fraudulent patterns: unusual amounts, times, locations
            typical_amount_range = (200, 2000)
            time_distribution = 'uniform'  # Active at unusual hours
            location_consistency = 0.3  # Low location consistency
        else:
            # Normal patterns: regular amounts, business hours, consistent locations
            typical_amount_range = (10, 500)
            time_distribution = 'business_hours'
            location_consistency = 0.8  # High location consistency
            
        user_locations = ['New York', 'Los Angeles', 'Chicago']
        preferred_location = random.choice(user_locations)
        
        for i in range(num_transactions):
            # Generate timestamp
            if time_distribution == 'business_hours':
                # More transactions during business hours
                hour = np.random.choice(range(24), p=self._get_business_hour_probabilities())
            else:
                hour = random.randint(0, 23)
                
            timestamp = base_time + timedelta(
                days=int(random.randint(0, 30)),
                hours=int(hour),
                minutes=int(random.randint(0, 59))
            )
            
            # Generate amount
            if is_fraudulent and random.random() < 0.3:
                # Occasionally very high amounts for fraud
                amount = random.uniform(1000, 5000)
            else:
                amount = random.uniform(*typical_amount_range)
                
            # Round to 2 decimal places, but fraudulent transactions might have unusual precision
            if is_fraudulent and random.random() < 0.2:
                amount = round(amount, 3)  # Unusual precision
            else:
                amount = round(amount, 2)
                
            # Generate merchant
            category = random.choice(self.merchant_categories)
            merchant = random.choice(self.merchants[category])
            
            # Generate location
            if random.random() < location_consistency:
                location = preferred_location
            else:
                location = random.choice(user_locations)
                
            # Fraudulent transactions might have no location or suspicious locations
            if is_fraudulent and random.random() < 0.2:
                location = random.choice(['', 'Unknown', 'Foreign'])
                
            transaction = {
                'transaction_id': f"{user_id}_txn_{i}",
                'user_id': user_id,
                'amount': amount,
                'timestamp': timestamp,
                'merchant_category': category,
                'merchant_name': merchant,
                'location': location,
                'label': 'fraud' if is_fraudulent else 'legitimate'
            }
            
            transactions.append(transaction)
            
        return transactions
    
    def _get_business_hour_probabilities(self) -> np.ndarray:
        """Get probability distribution favoring business hours"""
        probs = np.ones(24) * 0.02  # Base probability
        
        # Higher probability during business hours
        for hour in range(9, 18):
            probs[hour] = 0.08
            
        # Moderate probability during evening hours
        for hour in range(18, 22):
            probs[hour] = 0.04
            
        # Very low probability during night hours
        for hour in range(0, 6):
            probs[hour] = 0.005
            
        # Normalize
        probs = probs / probs.sum()
        return probs
    
    def add_fraudulent_patterns_to_legitimate_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add some fraudulent transactions to otherwise legitimate users"""
        df_copy = df.copy()
        
        # Select 10% of legitimate users to have fraudulent transactions
        legitimate_users = df_copy[df_copy['label'] == 'legitimate']['user_id'].unique()
        compromised_users = np.random.choice(legitimate_users, size=int(len(legitimate_users) * 0.1), replace=False)
        
        fraudulent_additions = []
        
        for user_id in compromised_users:
            # Add 1-3 fraudulent transactions per compromised user
            num_fraud_txns = random.randint(1, 3)
            fraud_transactions = self._generate_user_transactions(user_id, num_fraud_txns, is_fraudulent=True)
            
            # Update transaction IDs to avoid conflicts
            for i, txn in enumerate(fraud_transactions):
                txn['transaction_id'] = f"{user_id}_fraud_{i}"
                
            fraudulent_additions.extend(fraud_transactions)
        
        # Add fraudulent transactions
        fraud_df = pd.DataFrame(fraudulent_additions)
        combined_df = pd.concat([df_copy, fraud_df], ignore_index=True)
        
        return combined_df
    
    def collect_all_transaction_data(self, num_legitimate_users: int = 500, num_fraud_users: int = 50) -> pd.DataFrame:
        """Collect all transaction data"""
        print("Generating transaction dataset...")
        
        # Generate legitimate transactions
        legitimate_df = self.generate_legitimate_transactions(
            num_users=num_legitimate_users, 
            transactions_per_user=random.randint(30, 80)
        )
        
        # Generate purely fraudulent users
        fraudulent_df = self.generate_fraudulent_transactions(
            num_users=num_fraud_users, 
            transactions_per_user=random.randint(5, 20)
        )
        
        # Combine datasets
        combined_df = pd.concat([legitimate_df, fraudulent_df], ignore_index=True)
        
        # Add some fraudulent transactions to legitimate users (account takeover scenario)
        final_df = self.add_fraudulent_patterns_to_legitimate_users(combined_df)
        
        # Shuffle the data
        final_df = final_df.sample(frac=1).reset_index(drop=True)
        
        # Save raw data
        output_file = self.data_dir / "raw_transactions.csv"
        final_df.to_csv(output_file, index=False)
        print(f"Saved {len(final_df)} transactions to {output_file}")
        print(f"Fraud rate: {(final_df['label'] == 'fraud').mean():.2%}")
        
        return final_df

# Main collection script
def collect_all_datasets():
    """Collect all datasets"""
    base_dir = Path("ml_models/data")
    
    # Create collectors
    url_collector = URLDataCollector(base_dir / "raw" / "urls")
    email_collector = EmailDataCollector(base_dir / "raw" / "emails") 
    transaction_collector = TransactionDataCollector(base_dir / "raw" / "transactions")
    
    print("=== Collecting URL Data ===")
    url_data = url_collector.collect_all_url_data()
    
    print("\n=== Collecting Email Data ===")
    email_data = email_collector.collect_all_email_data()
    
    print("\n=== Collecting Transaction Data ===")
    transaction_data = transaction_collector.collect_all_transaction_data()
    
    print("\n=== Data Collection Summary ===")
    print(f"URLs collected: {len(url_data)}")
    print(f"Emails collected: {len(email_data)}")
    print(f"Transactions collected: {len(transaction_data)}")
    
    return url_data, email_data, transaction_data

if __name__ == "__main__":
    collect_all_datasets()