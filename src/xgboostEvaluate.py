
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, log_loss
#import shap
    

def evaluate_predictions(y_true, y_pred, y_pred_proba=None, model=None, X_data=None, 
                        compute_shap=False, feature_names=None, verbose=True):
    """
    Evaluate predictions with various metrics including SHAP values.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    y_pred_proba : numpy.ndarray, optional
        Predicted probabilities (for additional metrics)
    model : trained model, optional
        Model for SHAP computation
    X_data : numpy.ndarray, optional
        Input data for SHAP computation
    compute_shap : bool, default=False
        Whether to compute SHAP values
    feature_names : list, optional
        Names of features for SHAP analysis
    verbose : bool, default=True
        Whether to print evaluation results
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics and SHAP results
    """

    metrics = {}
    
    # Determine if it's classification or regression
    unique_true = np.unique(y_true)
    #unique_pred = np.unique(y_pred)
    is_classification = (len(unique_true) <= 50 and 
                        np.issubdtype(y_true.dtype, np.integer) and
                        np.issubdtype(y_pred.dtype, np.integer))
    
    if is_classification:
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if len(unique_true) == 2:  # Binary classification
            metrics['precision'] = precision_score(y_true, y_pred)
            metrics['recall'] = recall_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred)
            
            if y_pred_proba is not None:
                
                if y_pred_proba.ndim > 1:
                    proba_pos = y_pred_proba[:, 1]
                else:
                    proba_pos = y_pred_proba
                metrics['roc_auc'] = roc_auc_score(y_true, proba_pos)
                metrics['log_loss'] = log_loss(y_true, proba_pos)
        else:  # Multi-class
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        if verbose:
            print("\n=== Classification Results ===")
            for metric, value in metrics.items():
                print(f"{metric.upper()}: {value:.4f}")
            print("\nDetailed Classification Report:")
            print(classification_report(y_true, y_pred))
            
    else:
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        if verbose:
            print("\n=== Regression Results ===")
            for metric, value in metrics.items():
                print(f"{metric.upper()}: {value:.4f}")

    """
    Way to use function

    metrics = evaluate_predictions(y_test, y_pred)
    """
    
    return metrics

