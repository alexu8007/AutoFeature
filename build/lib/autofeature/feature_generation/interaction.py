"""
Interaction Transformer

This module provides transformers for creating interaction features between columns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from itertools import combinations
from autofeature.feature_generation.base import BaseTransformer


class InteractionTransformer(BaseTransformer):
    """Transformer for creating interaction features between columns.
    
    This transformer creates new features by combining existing ones through
    operations like multiplication, division, addition, and subtraction.
    """
    
    SUPPORTED_INTERACTIONS = {
        'multiplication': ('mult', lambda x, y: x * y),
        'division': ('div', lambda x, y: x / (y + np.finfo(float).eps)),
        'addition': ('add', lambda x, y: x + y),
        'subtraction': ('sub', lambda x, y: x - y),
        'ratio': ('ratio', lambda x, y: x / (y + np.finfo(float).eps)),
        'difference_ratio': ('diff_ratio', lambda x, y: (x - y) / (x + y + np.finfo(float).eps)),
    }
    
    def __init__(self, interaction_types: List[str] = None, max_combinations: int = 2,
                 exclude_columns: List[str] = None, include_columns: List[str] = None,
                 max_features: int = 100):
        """Initialize the interaction transformer.
        
        Args:
            interaction_types: Types of interactions to create
            max_combinations: Maximum number of columns to combine
            exclude_columns: Columns to exclude from transformation
            include_columns: Only include these columns in transformations
            max_features: Maximum number of features to generate
        """
        super().__init__()
        self.interaction_types = interaction_types or ['multiplication']
        self.max_combinations = max_combinations
        self.exclude_columns = exclude_columns or []
        self.include_columns = include_columns
        self.max_features = max_features
        self.numerical_columns = []
        
        # Validate interaction types
        for interaction in self.interaction_types:
            if interaction not in self.SUPPORTED_INTERACTIONS:
                raise ValueError(f"Interaction '{interaction}' is not supported. "
                                f"Supported interactions are: {list(self.SUPPORTED_INTERACTIONS.keys())}.")
                
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'InteractionTransformer':
        """Fit the transformer to the data by identifying numerical columns.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            self: The fitted transformer
        """
        self._validate_input(X)
        
        # Find numerical columns
        self.numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
        
        # Apply column filters
        if self.include_columns:
            self.numerical_columns = [col for col in self.numerical_columns if col in self.include_columns]
        
        self.numerical_columns = [col for col in self.numerical_columns if col not in self.exclude_columns]
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data by creating interaction features.
        
        Args:
            X: Input features
            
        Returns:
            pd.DataFrame: Transformed data with new features
        """
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted yet. Call fit or fit_transform first.")
        
        self._validate_input(X)
        result = X.copy()
        
        feature_count = 0
        
        # Generate combinations of columns
        for r in range(2, self.max_combinations + 1):
            if feature_count >= self.max_features:
                break
                
            for cols in combinations(self.numerical_columns, r):
                if feature_count >= self.max_features:
                    break
                    
                # Make sure all columns exist in the DataFrame
                if not all(col in X.columns for col in cols):
                    continue
                    
                # Apply each interaction type
                for interaction_name in self.interaction_types:
                    if feature_count >= self.max_features:
                        break
                        
                    suffix, interact_func = self.SUPPORTED_INTERACTIONS[interaction_name]
                    
                    # For interactions with more than 2 columns, apply pairwise
                    for col1, col2 in combinations(cols, 2):
                        if feature_count >= self.max_features:
                            break
                            
                        # Skip if any column has invalid values for this interaction
                        if not self._is_interaction_applicable(interaction_name, X[col1].values, X[col2].values):
                            continue
                            
                        # Generate the new feature name
                        new_col_name = f"{col1}_{suffix}_{col2}"
                        
                        # Apply the interaction
                        try:
                            result[new_col_name] = interact_func(X[col1].values, X[col2].values)
                            
                            # Register the feature
                            self._register_feature(new_col_name, {
                                'source_columns': [col1, col2],
                                'interaction': interaction_name,
                                'type': 'interaction'
                            })
                            
                            feature_count += 1
                        except Exception as e:
                            # Skip on error
                            continue
        
        return result
    
    def _is_interaction_applicable(self, interaction_name: str, data1: np.ndarray, data2: np.ndarray) -> bool:
        """Check if the interaction is applicable to the data.
        
        Args:
            interaction_name: Name of the interaction
            data1: First column data
            data2: Second column data
            
        Returns:
            bool: Whether the interaction is applicable
        """
        # Skip interactions that would produce NaN or infinite values
        if interaction_name in ['division', 'ratio'] and np.any(data2 == 0):
            return False
        if interaction_name == 'difference_ratio' and np.any((data1 + data2) == 0):
            return False
            
        return True 