# ml_models/training/train_url.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import MODEL_CONFIG, MODELS_DIR, SCALERS_DIR
from preprocessing.url_processor import URLFeatureExtractor, preprocess_url_dataset
from evaluation.metrics import ModelEvaluator

class URLModelTrainer:
    """Train URL phishing detection models"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = MODEL_CONFIG['url']['features']
        self.evaluator = ModelEvaluator()
        
    def load_and_preprocess_data(self, data_path: str) -> tuple:
        """Load and preprocess URL data"""
        print("Loading URL dataset...")
        
        # Check if processed data exists
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # If raw data needs processing
            raw_data_path = data_path.replace('processed', 'raw')
            if os.path.exists(raw_data_path):
                processed_path = data_path
                preprocess_url_dataset(raw_data_path, processed_path)
                df = pd.read_csv(processed_path)
            else:
                raise FileNotFoundError(f"No data found at {data_path} or {raw_data_path}")
        
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        
        # Prepare features and labels
        feature_columns = [col for col in self.feature_names if col in df.columns]
        if not feature_columns:
            # If processed features don't exist, extract them
            extractor = URLFeatureExtractor()
            feature_df = extractor.process_batch(df['url'].tolist())
            df = pd.merge(df, feature_df, on='url', how='inner')
            feature_columns = [col for col in self.feature_names if col in df.columns]
        
        X = df[feature_columns]
        y = (df['label'] == 'phishing').astype(int)  # Convert to binary
        
        return X, y
    
    def train_xgboost_model(self, X_train, X_val, y_train, y_val) -> xgb.XGBClassifier:
        """Train XGBoost model with hyperparameter tuning"""
        print("Training XGBoost model...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Base model
        base_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best XGBoost parameters: {grid_search.best_params_}")
        print(f"Best XGBoost CV score: {grid_search.best_score_:.4f}")
        
        # Train final model with best parameters
        best_model = grid_search.best_estimator_
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return best_model
    
    def train_random_forest_model(self, X_train, y_train) -> RandomForestClassifier:
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        # Parameter grid for Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best Random Forest parameters: {grid_search.best_params_}")
        print(f"Best Random Forest CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_and_evaluate_models(self, data_path: str):
        """Train and evaluate all models"""
        # Load data
        X, y = self.load_and_preprocess_data(data_path)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        print(f"Train set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost (primary model)
        xgb_model = self.train_xgboost_model(X_train_scaled, X_val_scaled, y_train, y_val)
        self.models['xgboost'] = xgb_model
        
        # Train Random Forest (backup model)
        rf_model = self.train_random_forest_model(X_train_scaled, y_train)
        self.models['random_forest'] = rf_model
        
        # Evaluate models
        print("\n=== Model Evaluation ===")
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n--- {model_name.upper()} Results ---")
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = self.evaluator.calculate_all_metrics(y_test, y_pred, y_pred_proba)
            results[model_name] = metrics
            
            # Print results
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Classification report
            print(f"\nClassification Report for {model_name}:")
            print(classification_report(y_test, y_pred))
        
        # Save best model (based on ROC-AUC)
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        best_model = self.models[best_model_name]
        
        print(f"\nBest model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
        
        # Save models and scaler
        self.save_models(best_model, best_model_name)
        
        # Generate evaluation report
        self.generate_evaluation_report(results, X_test_scaled, y_test)
        
        return results
    
    def save_models(self, best_model, model_name):
        """Save trained models and preprocessing artifacts"""
        # Create directories
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        SCALERS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save best model
        model_path = MODELS_DIR / MODEL_CONFIG['url']['model_file']
        joblib.dump(best_model, model_path)
        print(f"Saved best model ({model_name}) to {model_path}")
        
        # Save scaler
        scaler_path = SCALERS_DIR / MODEL_CONFIG['url']['scaler_file']
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
        
        # Save feature names
        feature_path = MODELS_DIR / "url_features.joblib"
        joblib.dump(self.feature_names, feature_path)
        print(f"Saved feature names to {feature_path}")
    
    def generate_evaluation_report(self, results, X_test, y_test):
        """Generate comprehensive evaluation report"""
        report_dir = Path("ml_models/evaluation/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC curves
        ax = axes[0, 0]
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = results[model_name]['roc_auc']
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - URL Classification')
        ax.legend()
        ax.grid(True)
        
        # Feature importance for best model
        ax = axes[0, 1]
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        best_model = self.models[best_model_name]
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Top 10 features
            
            ax.bar(range(len(indices)), importances[indices])
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Importance')
            ax.set_title(f'Top 10 Feature Importances - {best_model_name}')
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
        
        # Confusion matrix
        ax = axes[1, 0]
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Model comparison
        ax = axes[1, 1]
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        x_pos = np.arange(len(metrics_to_plot))
        width = 0.35
        
        for i, (model_name, metrics) in enumerate(results.items()):
            values = [metrics[metric] for metric in metrics_to_plot]
            ax.bar(x_pos + i * width, values, width, label=model_name)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x_pos + width/2)
        ax.set_xticklabels(metrics_to_plot, rotation=45)
        ax.legend()
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(report_dir / "url_model_evaluation.png", dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to {report_dir / 'url_model_evaluation.png'}")
        
        # Save detailed results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(report_dir / "url_model_results.csv")
        print(f"Detailed results saved to {report_dir / 'url_model_results.csv'}")

if __name__ == "__main__":
    trainer = URLModelTrainer()
    
    # Use processed data path
    data_path = "ml_models/data/processed/urls/processed_urls.csv"
    
    # Train models
    results = trainer.train_and_evaluate_models(data_path)
    
    print("\n=== Training Complete ===")
    for model_name, metrics in results.items():
        print(f"{model_name}: ROC-AUC = {metrics['roc_auc']:.4f}")

# ml_models/training/train_email.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import MODEL_CONFIG, MODELS_DIR, VECTORIZERS_DIR
from preprocessing.email_processor import EmailFeatureExtractor, EmailTextProcessor, preprocess_email_dataset
from evaluation.metrics import ModelEvaluator

class EmailModelTrainer:
    """Train email phishing/spam detection models"""
    
    def __init__(self):
        self.models = {}
        self.text_processor = EmailTextProcessor(max_features=MODEL_CONFIG['email']['max_features'])
        self.feature_extractor = EmailFeatureExtractor()
        self.evaluator = ModelEvaluator()
        
    def load_and_preprocess_data(self, data_path: str) -> tuple:
        """Load and preprocess email data"""
        print("Loading email dataset...")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            raw_data_path = data_path.replace('processed', 'raw')
            if os.path.exists(raw_data_path):
                vectorizer_path = VECTORIZERS_DIR / MODEL_CONFIG['email']['vectorizer_file']
                preprocess_email_dataset(raw_data_path, data_path, str(vectorizer_path))
                df = pd.read_csv(data_path)
            else:
                raise FileNotFoundError(f"No data found at {data_path} or {raw_data_path}")
        
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        
        # Separate TF-IDF features from manual features
        tfidf_columns = [col for col in df.columns if col.startswith('tfidf_')]
        manual_feature_columns = [col for col in df.columns if not col.startswith('tfidf_') and col not in ['email_id', 'label']]
        
        X_tfidf = df[tfidf_columns] if tfidf_columns else pd.DataFrame()
        X_manual = df[manual_feature_columns] if manual_feature_columns else pd.DataFrame()
        
        # Combine features
        X = pd.concat([X_tfidf, X_manual], axis=1)
        y = (df['label'] == 'phishing').astype(int)
        
        return X, y
    
    def train_logistic_regression(self, X_train, y_train) -> LogisticRegression:
        """Train Logistic Regression model"""
        print("Training Logistic Regression model...")
        
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        base_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best Logistic Regression parameters: {grid_search.best_params_}")
        print(f"Best Logistic Regression CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_random_forest(self, X_train, y_train) -> RandomForestClassifier:
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5]
        }
        
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best Random Forest parameters: {grid_search.best_params_}")
        print(f"Best Random Forest CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_and_evaluate_models(self, data_path: str):
        """Train and evaluate all models"""
        # Load data
        X, y = self.load_and_preprocess_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Train set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train models
        lr_model = self.train_logistic_regression(X_train, y_train)
        rf_model = self.train_random_forest(X_train, y_train)
        
        self.models['logistic_regression'] = lr_model
        self.models['random_forest'] = rf_model
        
        # Evaluate models
        print("\n=== Model Evaluation ===")
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n--- {model_name.upper()} Results ---")
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = self.evaluator.calculate_all_metrics(y_test, y_pred, y_pred_proba)
            results[model_name] = metrics
            
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            print(f"\nClassification Report for {model_name}:")
            print(classification_report(y_test, y_pred))
        
        # Save best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        best_model = self.models[best_model_name]
        
        print(f"\nBest model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
        
        self.save_models(best_model, best_model_name)
        self.generate_evaluation_report(results, X_test, y_test)
        
        return results
    
    def save_models(self, best_model, model_name):
        """Save trained models"""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        VECTORIZERS_DIR.mkdir(parents=True, exist_ok=True)
        
        model_path = MODELS_DIR / MODEL_CONFIG['email']['model_file']
        joblib.dump(best_model, model_path)
        print(f"Saved best model ({model_name}) to {model_path}")
    
    def generate_evaluation_report(self, results, X_test, y_test):
        """Generate evaluation report"""
        report_dir = Path("ml_models/evaluation/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(report_dir / "email_model_results.csv")
        print(f"Results saved to {report_dir / 'email_model_results.csv'}")

# ml_models/training/train_transaction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import MODEL_CONFIG, MODELS_DIR, SCALERS_DIR
from preprocessing.transaction_processor import TransactionDataProcessor, preprocess_transaction_dataset
from evaluation.metrics import ModelEvaluator

class TransactionModelTrainer:
    """Train transaction fraud detection models"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.processor = TransactionDataProcessor()
        self.evaluator = ModelEvaluator()
        
    def load_and_preprocess_data(self, data_path: str) -> tuple:
        """Load and preprocess transaction data"""
        print("Loading transaction dataset...")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            raw_data_path = data_path.replace('processed', 'raw')
            if os.path.exists(raw_data_path):
                scaler_path = SCALERS_DIR / MODEL_CONFIG['transaction']['scaler_file']
                encoders_path = SCALERS_DIR / "transaction_encoders.pkl"
                preprocess_transaction_dataset(raw_data_path, data_path, str(scaler_path), str(encoders_path))
                df = pd.read_csv(data_path)
            else:
                raise FileNotFoundError(f"No data found at {data_path} or {raw_data_path}")
        
        print(f"Dataset shape: {df.shape}")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        print(f"Fraud rate: {(df['label'] == 'fraud').mean():.2%}")
        
        # Prepare features
        feature_columns = [col for col in MODEL_CONFIG['transaction']['features'] if col in df.columns]
        X = df[feature_columns]
        y = (df['label'] == 'fraud').astype(int)
        
        return X, y
    
    def train_lightgbm_model(self, X_train, X_val, y_train, y_val) -> lgb.LGBMClassifier:
        """Train LightGBM model"""
        print("Training LightGBM model...")
        
        # Handle class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15],
            'num_leaves': [31, 63, 127]
        }
        
        base_model = lgb.LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best LightGBM parameters: {grid_search.best_params_}")
        print(f"Best LightGBM CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_random_forest_model(self, X_train, y_train) -> RandomForestClassifier:
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best Random Forest parameters: {grid_search.best_params_}")
        print(f"Best Random Forest CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_and_evaluate_models(self, data_path: str):
        """Train and evaluate all models"""
        # Load data
        X, y = self.load_and_preprocess_data(data_path)
        
        # Split data with stratification for imbalanced dataset
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        
        print(f"Train set size: {len(X_train)} (fraud: {y_train.sum()})")
        print(f"Validation set size: {len(X_val)} (fraud: {y_val.sum()})")
        print(f"Test set size: {len(X_test)} (fraud: {y_test.sum()})")
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        lgb_model = self.train_lightgbm_model(X_train_scaled, X_val_scaled, y_train, y_val)
        rf_model = self.train_random_forest_model(X_train_scaled, y_train)
        
        self.models['lightgbm'] = lgb_model
        self.models['random_forest'] = rf_model
        
        # Evaluate models
        print("\n=== Model Evaluation ===")
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n--- {model_name.upper()} Results ---")
            
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            metrics = self.evaluator.calculate_all_metrics(y_test, y_pred, y_pred_proba)
            results[model_name] = metrics
            
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            print(f"\nClassification Report for {model_name}:")
            print(classification_report(y_test, y_pred))
        
        # Save best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
        best_model = self.models[best_model_name]
        
        print(f"\nBest model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")
        
        self.save_models(best_model, best_model_name)
        self.generate_evaluation_report(results, X_test_scaled, y_test)
        
        return results
    
    def save_models(self, best_model, model_name):
        """Save trained models and preprocessing artifacts"""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        SCALERS_DIR.mkdir(parents=True, exist_ok=True)
        
        model_path = MODELS_DIR / MODEL_CONFIG['transaction']['model_file']
        joblib.dump(best_model, model_path)
        print(f"Saved best model ({model_name}) to {model_path}")
        
        scaler_path = SCALERS_DIR / MODEL_CONFIG['transaction']['scaler_file']
        joblib.dump(self.scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
    
    def generate_evaluation_report(self, results, X_test, y_test):
        """Generate evaluation report"""
        report_dir = Path("ml_models/evaluation/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame(results).T
        results_df.to_csv(report_dir / "transaction_model_results.csv")
        print(f"Results saved to {report_dir / 'transaction_model_results.csv'}")

if __name__ == "__main__":
    # Train all models
    models = ['url', 'email', 'transaction']
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*50}")
        
        if model_type == 'url':
            trainer = URLModelTrainer()
            data_path = "ml_models/data/processed/urls/processed_urls.csv"
        elif model_type == 'email':
            trainer = EmailModelTrainer()
            data_path = "ml_models/data/processed/emails/processed_emails.csv"
        else:  # transaction
            trainer = TransactionModelTrainer()
            data_path = "ml_models/data/processed/transactions/processed_transactions.csv"
        
        try:
            results = trainer.train_and_evaluate_models(data_path)
            print(f"\n{model_type.upper()} model training completed successfully!")
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
    
    print("\n" + "="*50)
    print("ALL MODEL TRAINING COMPLETED")
    print("="*50)