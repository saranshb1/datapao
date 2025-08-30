

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import gc
import psutil
import pickle
import os

def batchTrainXGBoost(X, y, batch_size=1000000, validation_size=0.2, 
                       xgb_params=None, early_stopping_rounds=10,
                       save_model_path=None, verbose=True, memory_limit_gb=None):
    """
    Train XGBoost model in batches for large datasets.
    
    Parameters:
    -----------
    X : numpy.ndarray or pandas.DataFrame
        Training features
    y : numpy.ndarray
        Training labels
    batch_size : int, default=1000000
        Size of each training batch
    validation_size : float, default=0.2
        Fraction of data to use for validation
    xgb_params : dict, optional
        XGBoost parameters. If None, uses optimized defaults for large datasets
    early_stopping_rounds : int, default=10
        Early stopping rounds for validation
    save_model_path : str, optional
        Path to save the trained model
    verbose : bool, default=True
        Whether to print progress information
    memory_limit_gb : float, optional
        Memory limit in GB. If specified, will adjust batch size accordingly
        
    Returns:
    --------
    model : xgb.XGBClassifier or xgb.XGBRegressor
        Trained XGBoost model
    training_history : dict
        Dictionary containing training metrics history
    """
    
    # Check available memory and adjust batch size if needed
    if memory_limit_gb:
        available_memory = psutil.virtual_memory().available / (1024**3)
        if verbose:
            print(f"Available memory: {available_memory:.2f} GB")
            print(f"Memory limit set to: {memory_limit_gb:.2f} GB")
        
        # Estimate memory usage per sample (rough estimation)
        bytes_per_sample = X.shape[1] * 8  # 8 bytes per float64
        max_samples = int((memory_limit_gb * 1024**3) / bytes_per_sample * 0.8)  # 80% safety margin
        batch_size = min(batch_size, max_samples)
        if verbose:
            print(f"Adjusted batch size to: {batch_size:,}")
    
    # Set default XGBoost parameters optimized for large datasets
    if xgb_params is None:
        # Determine if it's classification or regression based on target values
        unique_labels = np.unique(y)
        is_classification = len(unique_labels) <= 50 and np.issubdtype(y.dtype, np.integer)
        
        if is_classification:
            if len(unique_labels) == 2:
                objective = 'binary:logistic'
                eval_metric = 'logloss'
            else:
                objective = 'multi:softprob'
                eval_metric = 'mlogloss'
        else:
            objective = 'reg:squarederror'
            eval_metric = 'rmse'
            
        xgb_params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'tree_method': 'hist',
            'max_bin': 256,
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': 0
        }
        
        if is_classification and len(unique_labels) > 2:
            xgb_params['num_class'] = len(unique_labels)
    
    if verbose:
        print(f"Dataset shape: {X.shape}")
        print(f"Batch size: {batch_size:,}")
        print(f"Number of batches: {int(np.ceil(len(X) / batch_size))}")
        print(f"XGBoost parameters: {xgb_params}")
    
    # Create train/validation split
    if validation_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=42, stratify=y if len(np.unique(y)) < 50 else None
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None
    
    # Initialize model
    if xgb_params['objective'].startswith('reg'):
        model = xgb.XGBRegressor(**{k: v for k, v in xgb_params.items() if k != 'objective'})
    else:
        model = xgb.XGBClassifier(**{k: v for k, v in xgb_params.items() if k not in ['objective', 'num_class']})
    
    # Training history
    training_history = {
        'batch_scores': [],
        'val_scores': [],
        'memory_usage': []
    }
    
    # Calculate number of batches
    n_samples = len(X_train)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    # Initialize with first batch
    if verbose:
        print("\nStarting batch training...")
        
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        if verbose:
            print(f"Training batch {batch_idx + 1}/{n_batches} "
                  f"(samples {start_idx:,} to {end_idx:,})")
        
        try:
            if batch_idx == 0:
                # Fit the first batch
                if X_val is not None:
                    model.fit(
                        X_batch, y_batch,
                        eval_set=[(X_val, y_val)],
                        #early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                else:
                    model.fit(X_batch, y_batch)
            else:
                # For subsequent batches, we need to use warm_start or retrain
                # XGBoost doesn't have incremental learning, so we combine batches
                if batch_idx % 5 == 0 or batch_idx == n_batches - 1:  # Retrain every 5 batches or last batch
                    # Combine current batch with some previous data
                    prev_start = max(0, start_idx - batch_size * 2)
                    X_combined = X_train[prev_start:end_idx]
                    y_combined = y_train[prev_start:end_idx]
                    
                    if X_val is not None:
                        model.fit(
                            X_combined, y_combined,
                            eval_set=[(X_val, y_val)],
                            #early_stopping_rounds=early_stopping_rounds,
                            verbose=False
                        )
                    else:
                        model.fit(X_combined, y_combined)
            
            # Evaluate on validation set if available
            if X_val is not None:
                val_pred = model.predict(X_val)
                if xgb_params['objective'].startswith('reg'):
                    from sklearn.metrics import mean_squared_error
                    val_score = np.sqrt(mean_squared_error(y_val, val_pred))
                else:
                    val_score = accuracy_score(y_val, val_pred)
                training_history['val_scores'].append(val_score)
                
                if verbose:
                    print(f"  Validation score: {val_score:.4f}")
            
            # Track memory usage
            memory_usage = psutil.Process().memory_info().rss / (1024**3)
            training_history['memory_usage'].append(memory_usage)
            
            if verbose:
                print(f"  Memory usage: {memory_usage:.2f} GB")
            
            # Force garbage collection
            del X_batch, y_batch
            gc.collect()
            
        except Exception as e:
            print(f"Error in batch {batch_idx + 1}: {str(e)}")
            if "Invalid shape of labels" in str(e):
                print("Trying with reduced batch size...")
                # Recursively try with smaller batch size
                return batchTrainXGBoost(
                    X, y, batch_size=batch_size//2, 
                    validation_size=validation_size,
                    xgb_params=xgb_params, 
                    #early_stopping_rounds=early_stopping_rounds,
                    save_model_path=save_model_path, 
                    verbose=verbose, 
                    memory_limit_gb=memory_limit_gb
                )
            else:
                raise e
    
    # Final training on all data (or large subset if memory constrained)
    if verbose:
        print("\nPerforming final training on full dataset...")
    
    try:
        # Try training on full dataset
        if X_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                #early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
            
    except Exception as e:
        if verbose:
            print(f"Full dataset training failed: {str(e)}")
            print("Using model from batch training...")
    
    # Save model if path provided
    if save_model_path:
        model.save_model(save_model_path)
        if verbose:
            print(f"Model saved to: {save_model_path}")
    
    if verbose:
        print("Batch training completed!")
        if training_history['val_scores']:
            print(f"Best validation score: {max(training_history['val_scores']):.4f}")
        print(f"Peak memory usage: {max(training_history['memory_usage']):.2f} GB")

    """
    Way to use function:

    # Or with more control
    model, history = batchTrainXGBoost(
        X_train, y_train,
        batch_size=100000,            # Adjust based on your memory
        validation_size=0.1,          # No validation
        memory_limit_gb=6,            # Set based on your available RAM
        save_model_path='smallerBatchModel.json',
        verbose=True
        )

    # Check training progress
    print("Validation scores:", history['val_scores'])
    print("Memory usage over time:", history['memory_usage'])
    """

    
    return model, training_history



def batchPredictXGBoost(model, X_test, batch_size=1000000, predict_proba=False, 
                         memory_limit_gb=None, verbose=True):
    """
    Make predictions on large datasets in batches.
    
    Parameters:
    -----------
    model : trained XGBoost model
        The trained model to use for predictions
    X_test : numpy.ndarray or pandas.DataFrame
        Test features
    batch_size : int, default=1000000
        Size of each prediction batch
    predict_proba : bool, default=False
        Whether to return probabilities (for classification) or class predictions
    memory_limit_gb : float, optional
        Memory limit in GB. If specified, will adjust batch size accordingly
    verbose : bool, default=True
        Whether to print progress information
        
    Returns:
    --------
    predictions : numpy.ndarray
        Predictions for all test samples
    """
    
    # Check available memory and adjust batch size if needed
    if memory_limit_gb:
        available_memory = psutil.virtual_memory().available / (1024**3)
        if verbose:
            print(f"Available memory: {available_memory:.2f} GB")
            print(f"Memory limit set to: {memory_limit_gb:.2f} GB")
        
        # Estimate memory usage per sample
        bytes_per_sample = X_test.shape[1] * 8  # 8 bytes per float64
        max_samples = int((memory_limit_gb * 1024**3) / bytes_per_sample * 0.8)
        batch_size = min(batch_size, max_samples)
        if verbose:
            print(f"Adjusted batch size to: {batch_size:,}")
    
    n_samples = len(X_test)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    if verbose:
        print(f"Test dataset shape: {X_test.shape}")
        print(f"Prediction batch size: {batch_size:,}")
        print(f"Number of prediction batches: {n_batches}")
        print(f"Predict probabilities: {predict_proba}")
    
    predictions_list = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        X_batch = X_test[start_idx:end_idx]
        
        if verbose:
            print(f"Predicting batch {batch_idx + 1}/{n_batches} "
                  f"(samples {start_idx:,} to {end_idx:,})")
        
        try:
            if predict_proba:
                batch_pred = model.predict_proba(X_batch)
            else:
                batch_pred = model.predict(X_batch)
            
            predictions_list.append(batch_pred)
            
            # Track memory usage
            memory_usage = psutil.Process().memory_info().rss / (1024**3)
            if verbose:
                print(f"  Memory usage: {memory_usage:.2f} GB")
            
            # Force garbage collection
            del X_batch, batch_pred
            gc.collect()
            
        except Exception as e:
            print(f"Error in prediction batch {batch_idx + 1}: {str(e)}")
            # Try with smaller batch size
            if batch_size > 10000:
                print("Trying with reduced batch size...")
                return batchPredictXGBoost(
                    model, X_test, batch_size=batch_size//2, 
                    predict_proba=predict_proba, memory_limit_gb=memory_limit_gb, 
                    verbose=verbose
                )
            else:
                raise e
    
    # Combine all predictions
    if verbose:
        print("Combining predictions...")
    
    if predict_proba and len(predictions_list[0].shape) > 1:
        # For probability predictions (2D arrays)
        predictions = np.vstack(predictions_list)
    else:
        # For class predictions (1D arrays)
        predictions = np.hstack(predictions_list)
    
    if verbose:
        print(f"Final predictions shape: {predictions.shape}")
        print("Batch prediction completed!")

    """
    Way to use function

    y_pred = batch_predict_xgboost(
        model, X_test,
        batch_size=50000,
        predict_proba=False,  # Set to True for probabilities
        memory_limit_gb=6,
        verbose=True
    )

    """
    
    return predictions



