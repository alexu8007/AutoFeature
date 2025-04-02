"""
Input/Output utilities for the AutoFeature framework.

This module provides functions for loading and saving data in various formats,
as well as serialization utilities for storing models and configurations.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit


def load_data(file_path):
    """
    Load data from various file formats.
    
    Parameters
    ----------
    file_path : str
        Path to the data file (csv, excel, parquet, or pickle)
        
    Returns
    -------
    pandas.DataFrame
        Loaded data
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        if file_ext == '.csv':
            data = pd.read_csv(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            data = pd.read_excel(file_path)
        elif file_ext == '.parquet':
            data = pd.read_parquet(file_path)
        elif file_ext in ['.pkl', '.pickle']:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        return data
    
    except Exception as e:
        raise IOError(f"Error loading data from {file_path}: {str(e)}")


def save_data(data, file_path, **kwargs):
    """
    Save data to various file formats.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data to save
    file_path : str
        Path to save the data file (csv, excel, parquet, or pickle)
    **kwargs : dict
        Additional arguments for the save function
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    try:
        if file_ext == '.csv':
            data.to_csv(file_path, index=False, **kwargs)
        elif file_ext in ['.xls', '.xlsx']:
            data.to_excel(file_path, index=False, **kwargs)
        elif file_ext == '.parquet':
            data.to_parquet(file_path, index=False, **kwargs)
        elif file_ext in ['.pkl', '.pickle']:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    
    except Exception as e:
        raise IOError(f"Error saving data to {file_path}: {str(e)}")


def serialize_cv(cv):
    """
    Serialize cross-validation object to dictionary for JSON serialization.
    
    Parameters
    ----------
    cv : object
        Cross-validation object
        
    Returns
    -------
    dict
        Serialized cross-validation configuration
    """
    if isinstance(cv, int):
        return {"type": "kfold", "n_splits": cv}
    
    elif isinstance(cv, KFold):
        return {
            "type": "kfold",
            "n_splits": cv.n_splits,
            "shuffle": cv.shuffle,
            "random_state": cv.random_state
        }
    
    elif isinstance(cv, StratifiedKFold):
        return {
            "type": "stratified_kfold",
            "n_splits": cv.n_splits,
            "shuffle": cv.shuffle,
            "random_state": cv.random_state
        }
    
    elif isinstance(cv, TimeSeriesSplit):
        return {
            "type": "time_series_split",
            "n_splits": cv.n_splits,
            "max_train_size": cv.max_train_size,
            "test_size": cv.test_size
        }
    
    else:
        return {"type": "custom", "description": str(cv)}


def deserialize_cv(cv_config):
    """
    Deserialize cross-validation configuration to CV object.
    
    Parameters
    ----------
    cv_config : dict
        Serialized cross-validation configuration
        
    Returns
    -------
    object
        Cross-validation object
    """
    cv_type = cv_config.get("type", "kfold")
    
    if cv_type == "kfold":
        return KFold(
            n_splits=cv_config.get("n_splits", 5),
            shuffle=cv_config.get("shuffle", True),
            random_state=cv_config.get("random_state", None)
        )
    
    elif cv_type == "stratified_kfold":
        return StratifiedKFold(
            n_splits=cv_config.get("n_splits", 5),
            shuffle=cv_config.get("shuffle", True),
            random_state=cv_config.get("random_state", None)
        )
    
    elif cv_type == "time_series_split":
        return TimeSeriesSplit(
            n_splits=cv_config.get("n_splits", 5),
            max_train_size=cv_config.get("max_train_size", None),
            test_size=cv_config.get("test_size", None)
        )
    
    else:
        # Default to simple KFold
        return cv_config.get("n_splits", 5)


def save_json(data, file_path):
    """
    Save data to JSON file with proper handling of numpy types.
    
    Parameters
    ----------
    data : dict
        Data to save
    file_path : str
        Path to save the JSON file
    """
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return json.JSONEncoder.default(self, obj)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)
    except Exception as e:
        raise IOError(f"Error saving JSON to {file_path}: {str(e)}")


def load_json(file_path):
    """
    Load data from JSON file.
    
    Parameters
    ----------
    file_path : str
        Path to the JSON file
        
    Returns
    -------
    dict
        Loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Error loading JSON from {file_path}: {str(e)}")


def load_pipeline(file_path):
    """
    Load feature pipeline from pickle file.
    
    Parameters
    ----------
    file_path : str
        Path to the pickle file
        
    Returns
    -------
    FeaturePipeline
        Loaded feature pipeline
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pipeline file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline
    except Exception as e:
        raise IOError(f"Error loading pipeline from {file_path}: {str(e)}") 