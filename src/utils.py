
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    """
    Save a Python object to disk using joblib
    
    Parameters:
    -----------
    file_path : str
        Path where the object will be saved
    obj : object
        Any Python object to be saved
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path, compress=3)
    except Exception as e:
        print(f"Error saving object: {e}")

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return their performance metrics
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    models : dict
        Dictionary of models to evaluate {name: model}
        
    Returns:
    --------
    dict
        Dictionary of model names and their test R² scores
    """
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluate
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Save score to report
            report[model_name] = test_r2
            
            print(f"Model: {model_name}, Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            
        return report
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        return None

def load_object(filepath):
    """
    Load a Python object from disk using joblib
    
    Parameters:
    -----------
    filepath : str
        Path to the saved object
        
    Returns:
    --------
    object
        The loaded object
    """
    try:
        return joblib.load(filepath)
    except Exception as e:
        raise Exception(f"Error loading object from {filepath}: {e}")
