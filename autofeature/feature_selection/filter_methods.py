"""
Filter Methods for Feature Selection

This module provides filter-based methods for feature selection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from sklearn.feature_selection import (
    f_classif, mutual_info_classif, chi2,
    f_regression, mutual_info_regression
)
from scipy.stats import pearsonr, spearmanr
from autofeature.feature_selection.base import BaseSelector


class FilterSelector(BaseSelector):
    """Filter-based feature selector.
    
    This selector uses statistical measures to select features based on their
    individual relationship with the target variable.
    """
    
    SUPPORTED_METHODS = {
        'f_classif': f_classif,
        'mutual_info_classif': mutual_info_classif,
        'chi2': chi2,
        'f_regression': f_regression,
        'mutual_info_regression': mutual_info_regression,
        'pearson': lambda X, y: np.array([
            abs(pearsonr(X[col].values, y.values)[0]) if len(np.unique(X[col].values)) > 1 else 0
            for col in X.columns
        ]),
        'spearman': lambda X, y: np.array([
            abs(spearmanr(X[col].values, y.values)[0]) if len(np.unique(X[col].values)) > 1 else 0
            for col in X.columns
        ]),
        'variance': lambda X, y: np.array([
            np.var(X[col].values) if len(np.unique(X[col].values)) > 1 else 0
            for col in X.columns
        ])
    }
    
    def __init__(self, method: str = 'f_regression', 
                 k: Union[int, float] = 10, 
                 threshold: Optional[float] = None,
                 custom_score_func: Optional[Callable] = None,
                 categorical_features: Optional[List[str]] = None,
                 handle_na: str = 'drop'):
        """Initialize the filter selector.
        
        Args:
            method: Method for scoring features
            k: Number of features to select (if int) or fraction (if float < 1.0)
            threshold: Minimum score for feature selection (if provided, overrides k)
            custom_score_func: Custom score function (if provided, overrides method)
            categorical_features: List of categorical features for special handling
            handle_na: Strategy for handling missing values ('drop', 'mean', 'median', 'mode')
        """
        super().__init__()
        self.method = method
        self.k = k
        self.threshold = threshold
        self.custom_score_func = custom_score_func
        self.categorical_features = categorical_features or []
        self.handle_na = handle_na
        
        # Validate method
        if self.method not in self.SUPPORTED_METHODS and not self.custom_score_func:
            raise ValueError(f"Method '{method}' is not supported. "
                            f"Supported methods are: {list(self.SUPPORTED_METHODS.keys())}. "
                            f"Alternatively, provide a custom score function.")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FilterSelector':
        """Fit the selector to the data by scoring features.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            self: The fitted selector
        """
        self._validate_input(X)
        
        # Preprocess the data
        X_processed, y_processed = self._preprocess_data(X, y)
        
        # Handle empty data
        if X_processed.shape[0] == 0 or X_processed.shape[1] == 0:
            self.selected_features = []
            self.feature_importances_ = {}
            self.is_fitted = True
            return self
        
        # Check for regression or classification task
        is_regression = self._is_regression_task(y_processed)
        
        # Apply the appropriate method based on task type
        if self.custom_score_func:
            scores = self.custom_score_func(X_processed, y_processed)
        else:
            score_func = self._get_score_func(is_regression)
            
            try:
                # Calculate feature scores
                scores, _ = score_func(X_processed, y_processed)
            except Exception:
                # Fallback to a simpler method if the chosen one fails
                if is_regression:
                    scores, _ = f_regression(X_processed, y_processed)
                else:
                    scores, _ = f_classif(X_processed, y_processed)
        
        # Handle NaN scores
        scores = np.nan_to_num(scores)
        
        # Create feature importance dictionary
        self.feature_importances_ = dict(zip(X_processed.columns, scores))
        
        # Select features
        if self.threshold is not None:
            # Select by threshold
            self.selected_features = [
                col for col, score in self.feature_importances_.items()
                if score >= self.threshold
            ]
        else:
            # Select by number of features
            k = self.k
            if isinstance(k, float) and k < 1.0:
                k = max(1, int(k * len(self.feature_importances_)))
                
            # Sort features by importance
            sorted_features = sorted(
                self.feature_importances_.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select top k features
            self.selected_features = [f[0] for f in sorted_features[:k]]
        
        self.is_fitted = True
        return self
    
    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Preprocess the data for feature selection.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            tuple: Processed X and y
        """
        # Check for valid features
        self._check_input_features(X)
        
        # Make a copy to avoid modifying the original data
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Handle missing values
        if self.handle_na == 'drop':
            # Drop rows with missing values
            valid_idx = ~(X_processed.isna().any(axis=1) | y_processed.isna())
            X_processed = X_processed[valid_idx]
            y_processed = y_processed[valid_idx]
        else:
            # Fill missing values
            for col in X_processed.columns:
                if X_processed[col].isna().any():
                    if self.handle_na == 'mean':
                        X_processed[col].fillna(X_processed[col].mean(), inplace=True)
                    elif self.handle_na == 'median':
                        X_processed[col].fillna(X_processed[col].median(), inplace=True)
                    elif self.handle_na == 'mode':
                        X_processed[col].fillna(X_processed[col].mode()[0], inplace=True)
            
            # Drop rows where y is missing
            valid_idx = ~y_processed.isna()
            X_processed = X_processed[valid_idx]
            y_processed = y_processed[valid_idx]
        
        # Encode categorical features if needed
        for col in X_processed.columns:
            if col in self.categorical_features or X_processed[col].dtype == 'object':
                X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # Handle special cases for certain methods
        if self.method == 'chi2':
            # Chi2 requires non-negative values
            for col in X_processed.columns:
                if (X_processed[col] < 0).any():
                    X_processed[col] = X_processed[col] - X_processed[col].min()
        
        return X_processed, y_processed
    
    def _get_score_func(self, is_regression: bool) -> Callable:
        """Get the appropriate score function based on task type.
        
        Args:
            is_regression: Whether this is a regression task
            
        Returns:
            Callable: Score function
        """
        if self.method in self.SUPPORTED_METHODS:
            # Direct mapping for general methods
            if self.method in ['variance', 'pearson', 'spearman']:
                return self.SUPPORTED_METHODS[self.method]
                
            # Task specific methods
            if is_regression:
                if self.method in ['f_regression', 'mutual_info_regression']:
                    return self.SUPPORTED_METHODS[self.method]
                else:
                    # Default to f_regression if the method is not appropriate
                    return self.SUPPORTED_METHODS['f_regression']
            else:
                if self.method in ['f_classif', 'mutual_info_classif', 'chi2']:
                    return self.SUPPORTED_METHODS[self.method]
                else:
                    # Default to f_classif if the method is not appropriate
                    return self.SUPPORTED_METHODS['f_classif']
        else:
            # Default methods if the specified one is not found
            if is_regression:
                return self.SUPPORTED_METHODS['f_regression']
            else:
                return self.SUPPORTED_METHODS['f_classif']
    
    def _is_regression_task(self, y: pd.Series) -> bool:
        """Determine if this is a regression or classification task.
        
        Args:
            y: Target variable
            
        Returns:
            bool: True if regression, False if classification
        """
        unique_values = y.nunique()
        
        # Heuristic: if few unique values or object/category type, likely classification
        if unique_values <= 10 or y.dtype in ['object', 'category', 'bool']:
            return False
        
        # If float type with many unique values, likely regression
        if y.dtype.kind in 'fc' and unique_values > 10:
            return True
            
        # Default to classification
        return False 