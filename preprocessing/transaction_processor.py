# ml_models/preprocessing/transaction_processor.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

class TransactionFeatureExtractor:
    """Extract features from transaction data for fraud detection"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.high_risk_merchants = [
            'gas_station', 'atm', 'online', 'grocery', 'restaurant',
            'entertainment', 'travel', 'hotel', 'airline'
        ]
        
    def extract_features(self, transaction_data: Dict, user_history: pd.DataFrame = None) -> Dict[str, Union[int, float]]:
        """Extract all features from a single transaction"""
        features = {}
        
        try:
            # Basic transaction features
            features.update(self._extract_basic_features(transaction_data))
            
            # Temporal features
            features.update(self._extract_temporal_features(transaction_data))
            
            # Merchant features
            features.update(self._extract_merchant_features(transaction_data))
            
            # User behavioral features (if history provided)
            if user_history is not None:
                features.update(self._extract_user_features(transaction_data, user_history))
            else:
                features.update(self._get_default_user_features())
                
        except Exception as e:
            print(f"Error processing transaction: {e}")
            features = self._get_default_features()
            
        return features
    
    def _extract_basic_features(self, transaction_data: Dict) -> Dict[str, Union[int, float]]:
        """Extract basic transaction features"""
        amount = float(transaction_data.get('amount', 0))
        
        return {
            'amount': amount,
            'amount_log': np.log1p(amount),  # Log transform for skewed amounts
            'is_round_amount': 1 if amount == round(amount) else 0,
            'amount_decimal_places': len(str(amount).split('.')[-1]) if '.' in str(amount) else 0
        }
    
    def _extract_temporal_features(self, transaction_data: Dict) -> Dict[str, Union[int, float]]:
        """Extract time-based features"""
        timestamp = transaction_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_night': 1 if timestamp.hour >= 22 or timestamp.hour <= 6 else 0,
            'is_business_hours': 1 if 9 <= timestamp.hour <= 17 else 0,
            'day_of_month': timestamp.day,
            'month': timestamp.month
        }
    
    def _extract_merchant_features(self, transaction_data: Dict) -> Dict[str, Union[int, float]]:
        """Extract merchant-related features"""
        merchant_category = transaction_data.get('merchant_category', 'unknown').lower()
        merchant_name = transaction_data.get('merchant_name', '').lower()
        location = transaction_data.get('location', '')
        
        return {
            'is_high_risk_merchant': 1 if merchant_category in self.high_risk_merchants else 0,
            'is_online_transaction': 1 if 'online' in merchant_name or 'web' in merchant_name else 0,
            'merchant_category_encoded': self._encode_categorical(merchant_category, 'merchant_category'),
            'has_location': 1 if location else 0
        }
    
    def _extract_user_features(self, transaction_data: Dict, user_history: pd.DataFrame) -> Dict[str, float]:
        """Extract user behavioral features based on history"""
        user_id = transaction_data.get('user_id')
        current_amount = float(transaction_data.get('amount', 0))
        current_time = pd.to_datetime(transaction_data.get('timestamp', datetime.now()))
        
        # Filter user history
        user_txns = user_history[user_history['user_id'] == user_id].copy()
        
        if user_txns.empty:
            return self._get_default_user_features()
        
        # Calculate time-based features
        user_txns['timestamp'] = pd.to_datetime(user_txns['timestamp'])
        user_txns = user_txns.sort_values('timestamp')
        
        # Recent transaction patterns
        recent_1h = user_txns[user_txns['timestamp'] >= current_time - timedelta(hours=1)]
        recent_24h = user_txns[user_txns['timestamp'] >= current_time - timedelta(hours=24)]
        recent_7d = user_txns[user_txns['timestamp'] >= current_time - timedelta(days=7)]
        
        # Time since last transaction
        if not user_txns.empty:
            last_transaction_time = user_txns['timestamp'].max()
            time_since_last = (current_time - last_transaction_time).total_seconds() / 3600  # in hours
        else:
            time_since_last = 24  # Default to 24 hours
            
        # Amount-based features
        amounts_7d = recent_7d['amount'].astype(float)
        avg_amount_7d = amounts_7d.mean() if not amounts_7d.empty else current_amount
        std_amount_7d = amounts_7d.std() if len(amounts_7d) > 1 else 0
        
        # Z-score of current amount
        if std_amount_7d > 0:
            amount_zscore = abs(current_amount - avg_amount_7d) / std_amount_7d
        else:
            amount_zscore = 0
            
        # User account age
        first_transaction = user_txns['timestamp'].min()
        user_age_days = (current_time - first_transaction).days
        
        return {
            'user_age_days': user_age_days,
            'time_since_last_transaction': time_since_last,
            'transaction_frequency_1h': len(recent_1h),
            'transaction_frequency_24h': len(recent_24h),
            'transaction_frequency_7d': len(recent_7d),
            'avg_amount_7d': avg_amount_7d,
            'std_amount_7d': std_amount_7d,
            'amount_zscore': amount_zscore,
            'total_transactions': len(user_txns)
        }
    
    def _encode_categorical(self, value: str, column_name: str) -> int:
        """Encode categorical variables"""
        if column_name not in self.label_encoders:
            self.label_encoders[column_name] = LabelEncoder()
            # Fit with common values to avoid unseen category issues
            common_values = ['unknown', 'other', value]
            self.label_encoders[column_name].fit(common_values)
        
        try:
            return self.label_encoders[column_name].transform([value])[0]
        except ValueError:
            # Return code for 'unknown' if unseen category
            return self.label_encoders[column_name].transform(['unknown'])[0]
    
    def _get_default_user_features(self) -> Dict[str, float]:
        """Return default user features when no history is available"""
        return {
            'user_age_days': 0,
            'time_since_last_transaction': 24,
            'transaction_frequency_1h': 0,
            'transaction_frequency_24h': 0,
            'transaction_frequency_7d': 0,
            'avg_amount_7d': 0,
            'std_amount_7d': 0,
            'amount_zscore': 0,
            'total_transactions': 0
        }
    
    def _get_default_features(self) -> Dict[str, Union[int, float]]:
        """Return default feature values"""
        default_features = {
            'amount': 0, 'amount_log': 0, 'is_round_amount': 0, 'amount_decimal_places': 0,
            'hour': 12, 'day_of_week': 1, 'is_weekend': 0, 'is_night': 0,
            'is_business_hours': 1, 'day_of_month': 15, 'month': 6,
            'is_high_risk_merchant': 0, 'is_online_transaction': 0,
            'merchant_category_encoded': 0, 'has_location': 0
        }
        default_features.update(self._get_default_user_features())
        return default_features
    
    def process_batch(self, transactions: List[Dict], user_histories: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Process multiple transactions and return DataFrame"""
        features_list = []
        
        for transaction in transactions:
            user_id = transaction.get('user_id')
            user_history = user_histories.get(user_id) if user_histories else None
            
            features = self.extract_features(transaction, user_history)
            features['transaction_id'] = transaction.get('transaction_id', len(features_list))
            features_list.append(features)
            
        return pd.DataFrame(features_list)

class TransactionDataProcessor:
    """Main processor for transaction datasets"""
    
    def __init__(self):
        self.feature_extractor = TransactionFeatureExtractor()
        self.scaler = StandardScaler()
        self.numeric_features = [
            'amount', 'amount_log', 'user_age_days', 'time_since_last_transaction',
            'avg_amount_7d', 'std_amount_7d', 'amount_zscore'
        ]
        
    def create_user_histories(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create user transaction histories for feature engineering"""
        user_histories = {}
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        for user_id in df['user_id'].unique():
            user_data = df_sorted[df_sorted['user_id'] == user_id].copy()
            user_histories[user_id] = user_data
            
        return user_histories
    
    def add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add velocity-based features to dataset"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp'])
        
        # Calculate rolling statistics
        df['amount_rolling_mean_3'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['amount_rolling_std_3'] = df.groupby('user_id')['amount'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std().fillna(0)
        )
        
        # Transaction count in last 24h
        df['txn_count_24h'] = df.groupby('user_id').apply(
            lambda group: group.set_index('timestamp')['amount'].rolling('24H').count()
        ).reset_index(level=0, drop=True)
        
        return df
    
    def fit_preprocessors(self, df: pd.DataFrame) -> None:
        """Fit preprocessing components"""
        # Fit label encoders for categorical variables
        if 'merchant_category' in df.columns:
            unique_categories = df['merchant_category'].unique()
            self.feature_extractor.label_encoders['merchant_category'] = LabelEncoder()
            self.feature_extractor.label_encoders['merchant_category'].fit(unique_categories)
        
        # Extract features for scaling
        sample_transactions = df.head(1000).to_dict('records')
        features_df = self.feature_extractor.process_batch(sample_transactions)
        
        # Fit scaler on numeric features
        if not features_df.empty:
            numeric_cols = [col for col in self.numeric_features if col in features_df.columns]
            if numeric_cols:
                self.scaler.fit(features_df[numeric_cols])
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess entire dataset"""
        print("Adding velocity features...")
        df = self.add_velocity_features(df)
        
        print("Creating user histories...")
        user_histories = self.create_user_histories(df)
        
        print("Extracting transaction features...")
        # Process in batches to avoid memory issues
        batch_size = 1000
        processed_batches = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_transactions = batch.to_dict('records')
            
            # Create user histories for this batch
            batch_histories = {uid: user_histories[uid] for uid in batch['user_id'].unique() if uid in user_histories}
            
            features_df = self.feature_extractor.process_batch(batch_transactions, batch_histories)
            processed_batches.append(features_df)
            
            if i % 5000 == 0:
                print(f"Processed {i} transactions...")
        
        # Combine all batches
        final_df = pd.concat(processed_batches, ignore_index=True)
        
        # Add labels
        final_df = final_df.merge(
            df[['transaction_id', 'label']].reset_index().rename(columns={'index': 'transaction_id'}),
            on='transaction_id',
            how='left'
        )
        
        return final_df
    
    def save_preprocessors(self, scaler_file: str, encoders_file: str) -> None:
        """Save fitted preprocessors"""
        joblib.dump(self.scaler, scaler_file)
        joblib.dump(self.feature_extractor.label_encoders, encoders_file)
    
    def load_preprocessors(self, scaler_file: str, encoders_file: str) -> None:
        """Load fitted preprocessors"""
        self.scaler = joblib.load(scaler_file)
        self.feature_extractor.label_encoders = joblib.load(encoders_file)

def preprocess_transaction_dataset(input_file: str, output_file: str, scaler_file: str, encoders_file: str) -> None:
    """Preprocess transaction dataset and save artifacts"""
    # Load dataset
    df = pd.read_csv(input_file)
    
    # Initialize processor
    processor = TransactionDataProcessor()
    
    # Fit preprocessors
    processor.fit_preprocessors(df)
    
    # Preprocess dataset
    processed_df = processor.preprocess_dataset(df)
    
    # Save processed data and preprocessors
    processed_df.to_csv(output_file, index=False)
    processor.save_preprocessors(scaler_file, encoders_file)
    
    print(f"Processed transaction dataset saved to {output_file}")
    print(f"Scaler saved to {scaler_file}")
    print(f"Encoders saved to {encoders_file}")
    print(f"Dataset shape: {processed_df.shape}")

if __name__ == "__main__":
    # Example usage
    extractor = TransactionFeatureExtractor()
    
    test_transactions = [
        {
            'transaction_id': 1,
            'user_id': 'user_123',
            'amount': 50.00,
            'timestamp': '2024-01-15 14:30:00',
            'merchant_category': 'grocery',
            'merchant_name': 'SuperMart',
            'location': 'New York'
        },
        {
            'transaction_id': 2,
            'user_id': 'user_456',
            'amount': 2500.00,
            'timestamp': '2024-01-15 23:45:00',
            'merchant_category': 'online',
            'merchant_name': 'SuspiciousStore',
            'location': ''
        }
    ]
    
    for transaction in test_transactions:
        features = extractor.extract_features(transaction)
        print(f"\nTransaction: {transaction['amount']} at {transaction['merchant_name']}")
        print(f"Features: {features}")