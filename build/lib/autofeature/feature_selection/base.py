"""
Base Feature Selector

This module defines the base class for all feature selectors.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple


class BaseSelector(ABC):
    """Base class for all feature selectors.
    
    All feature selection methods should inherit from this class
    and implement the fit and transform methods.
    """
    
    def __init__(self, **kwargs):
        """Initialize the selector with configuration parameters."""
        self.is_fitted = False
        self.selected_features = []
        self.feature_importances_ = {}
        self._set_params(**kwargs)
    
    def _set_params(self, **kwargs):
        """Set parameters for the selector."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseSelector':
        """Fit the selector to the data.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            self: The fitted selector
        """
        pass
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data by selecting features.
        
        Args:
            X: Input features
            
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        if not self.is_fitted:
            raise ValueError("Selector is not fitted yet. Call fit or fit_transform first.")
        
        self._validate_input(X)
        
        # Return only the selected features
        selected_cols = [col for col in self.selected_features if col in X.columns]
        return X[selected_cols]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit the selector to the data and transform it.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        return self.fit(X, y).transform(X)
    
    def get_selected_features(self) -> List[str]:
        """Get the names of the selected features.
        
        Returns:
            List[str]: Names of the selected features
        """
        if not self.is_fitted:
            raise ValueError("Selector is not fitted yet. Call fit or fit_transform first.")
        return self.selected_features
    
    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dict: Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Selector is not fitted yet. Call fit or fit_transform first.")
        return self.feature_importances_
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate the input data.
        
        Args:
            X: Input features
            
        Raises:
            ValueError: If the input data is invalid
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
            
    def _check_input_features(self, X: pd.DataFrame) -> None:
        """Check that the input features are valid.
        
        Args:
            X: Input features
            
        Raises:
            ValueError: If the input features are invalid
        """
        # Check for constant columns
        constant_cols = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_cols.append(col)
                
        if constant_cols:
            # We don't throw an error, but we inform the user
            print(f"Warning: {len(constant_cols)} constant features detected: {constant_cols[:5]}...")
            
        # Check for columns with many NaN values
        nan_cols = []
        for col in X.columns:
            if X[col].isna().mean() > 0.9:  # More than 90% NaN
                nan_cols.append(col)
                
        if nan_cols:
            print(f"Warning: {len(nan_cols)} features with >90% missing values detected: {nan_cols[:5]}...")
    
    def _calculate_correlation_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate the correlation matrix for numerical features.
        
        Args:
            X: Input features
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Get numerical columns
        num_cols = X.select_dtypes(include=['number']).columns
        
        # Calculate correlation matrix
        corr_matrix = X[num_cols].corr().abs()
        
        return corr_matrix 