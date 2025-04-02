"""
Base Transformer

This module defines the base class for all feature transformers.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseTransformer(ABC):
    """Base class for all feature transformers.
    
    All feature generation transformers should inherit from this class
    and implement the fit and transform methods.
    """
    
    def __init__(self, **kwargs):
        """Initialize the transformer with configuration parameters."""
        self.is_fitted = False
        self.feature_metadata = {}
        self._set_params(**kwargs)
    
    def _set_params(self, **kwargs):
        """Set parameters for the transformer."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BaseTransformer':
        """Fit the transformer to the data.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            self: The fitted transformer
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data by generating new features.
        
        Args:
            X: Input features
            
        Returns:
            pd.DataFrame: Transformed data with new features
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the transformer to the data and transform it.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            pd.DataFrame: Transformed data with new features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the generated features.
        
        Returns:
            List[str]: Names of the generated features
        """
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted yet. Call fit or fit_transform first.")
        return list(self.feature_metadata.keys())
    
    def get_feature_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata about the generated features.
        
        Returns:
            Dict: Dictionary mapping feature names to their metadata
        """
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted yet. Call fit or fit_transform first.")
        return self.feature_metadata
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate the input data.
        
        Args:
            X: Input features
            
        Raises:
            ValueError: If the input data is invalid
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
            
    def _register_feature(self, name: str, metadata: Dict[str, Any]) -> None:
        """Register a new feature with metadata.
        
        Args:
            name: Name of the new feature
            metadata: Metadata about the feature
        """
        self.feature_metadata[name] = metadata 