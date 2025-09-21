# ml_models/evaluation/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import time

class ModelEvaluator:
    """Comprehensive model evaluation metrics and utilities"""
    
    def __init__(self):
        self.metrics_history = {}
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate all standard classification metrics
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels  
            y_pred_proba: Predicted probabilities for positive class
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC (handles edge cases)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['roc_auc'] = 0.5  # Random classifier performance
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # False Positive Rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Negative Rate  
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Positive/Negative Predictive Values
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # Matthews Correlation Coefficient
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics['mcc'] = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
        
        return metrics
    
    def calculate_threshold_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate metrics across different probability thresholds
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            thresholds: Array of thresholds to evaluate
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            metrics = self.calculate_all_metrics(y_true, y_pred, y_pred_proba)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              metric: str = 'f1_score') -> Tuple[float, float]:
        """
        Find optimal threshold based on specified metric
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1_score', 'precision', 'recall', etc.)
            
        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        threshold_metrics = self.calculate_threshold_metrics(y_true, y_pred_proba)
        
        best_row = threshold_metrics.loc[threshold_metrics[metric].idxmax()]
        return best_row['threshold'], best_row[metric]
    
    def evaluate_model_performance_by_class(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                          class_names: list = None) -> Dict:
        """
        Detailed evaluation by class
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names for classes
            
        Returns:
            Dictionary with per-class metrics
        """
        if class_names is None:
            class_names = ['Legitimate', 'Fraudulent']
        
        # Classification report as dict
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': class_names
        }
    
    def benchmark_inference_speed(self, model, X_test: np.ndarray, n_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference speed
        
        Args:
            model: Trained model with predict/predict_proba methods
            X_test: Test data
            n_iterations: Number of timing iterations
            
        Returns:
            Dictionary with timing statistics
        """
        # Single prediction timing
        single_times = []
        for _ in range(n_iterations):
            start_time = time.time()
            _ = model.predict(X_test[:1])
            single_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Batch prediction timing
        batch_times = []
        batch_sizes = [1, 10, 50, 100] if len(X_test) >= 100 else [1, 10, min(50, len(X_test))]
        
        for batch_size in batch_sizes:
            batch_time_list = []
            for _ in range(min(n_iterations, 20)):  # Fewer iterations for larger batches
                start_time = time.time()
                _ = model.predict(X_test[:batch_size])
                batch_time_list.append((time.time() - start_time) * 1000)
            batch_times.append({
                'batch_size': batch_size,
                'mean_time_ms': np.mean(batch_time_list),
                'std_time_ms': np.std(batch_time_list)
            })
        
        return {
            'single_prediction': {
                'mean_time_ms': np.mean(single_times),
                'std_time_ms': np.std(single_times),
                'min_time_ms': np.min(single_times),
                'max_time_ms': np.max(single_times),
                'p95_time_ms': np.percentile(single_times, 95)
            },
            'batch_predictions': batch_times
        }
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       model_name: str = "Model", save_path: str = None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Model", save_path: str = None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'{model_name}')
        
        # Baseline (random classifier performance)
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'Baseline (Positive Rate = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: list = None, save_path: str = None):
        """Plot confusion matrix heatmap"""
        if class_names is None:
            class_names = ['Legitimate', 'Fraudulent']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_summary(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_pred_proba: np.ndarray, model_name: str) -> Dict:
        """
        Generate comprehensive evaluation summary
        
        Returns:
            Dictionary with all evaluation results
        """
        # Basic metrics
        metrics = self.calculate_all_metrics(y_true, y_pred, y_pred_proba)
        
        # Threshold analysis
        optimal_threshold, optimal_f1 = self.find_optimal_threshold(y_true, y_pred_proba)
        
        # Per-class evaluation
        class_eval = self.evaluate_model_performance_by_class(y_true, y_pred)
        
        # Summary statistics
        summary = {
            'model_name': model_name,
            'test_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true)),
            'positive_rate': float(np.mean(y_true)),
            'metrics': metrics,
            'optimal_threshold': optimal_threshold,
            'optimal_f1_score': optimal_f1,
            'per_class_evaluation': class_eval,
            'evaluation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return summary
    
    def save_evaluation_report(self, evaluation_summary: Dict, file_path: str):
        """Save evaluation summary to JSON file"""
        import json
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Recursively convert numpy types
        def recursive_convert(data):
            if isinstance(data, dict):
                return {key: recursive_convert(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        converted_summary = recursive_convert(evaluation_summary)
        
        with open(file_path, 'w') as f:
            json.dump(converted_summary, f, indent=2)
        
        print(f"Evaluation report saved to {file_path}")

if __name__ == "__main__":
    # Example usage
    print("Testing ModelEvaluator...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
    y_pred_proba = np.random.beta(2, 5, n_samples)  # Random probabilities
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Calculate metrics
    metrics = evaluator.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    print("\nBasic Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = evaluator.find_optimal_threshold(y_true, y_pred_proba)
    print(f"\nOptimal threshold: {optimal_threshold:.3f} (F1: {optimal_f1:.4f})")
    
    # Generate full summary
    summary = evaluator.generate_evaluation_summary(y_true, y_pred, y_pred_proba, "Test Model")
    print(f"\nEvaluation Summary Generated for {summary['model_name']}")
    print(f"Test samples: {summary['test_samples']}")
    print(f"ROC-AUC: {summary['metrics']['roc_auc']:.4f}")
    
    print("\nModelEvaluator testing complete!")