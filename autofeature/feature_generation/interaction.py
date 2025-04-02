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
        
        # Dictionary to collect all new columns
        new_columns = {}
        
        # Filter out excluded columns
        allowed_cols = [c for c in X.columns if c not in self.exclude_columns]
        
        # Process each interaction type
        for interaction_type in self.interaction_types:
            interact_func = self._get_interaction_function(interaction_type)
            
            # Get potential column pairs for this interaction
            if self.column_pairs:
                column_pairs = [
                    (col1, col2) for col1, col2 in self.column_pairs 
                    if col1 in allowed_cols and col2 in allowed_cols
                ]
            else:
                # Generate column pairs based on data types
                numeric_cols = self._get_numeric_columns(X, allowed_cols)
                column_pairs = list(self._generate_column_pairs(numeric_cols))
            
            # Limit the number of pairs if required
            if self.max_features and len(column_pairs) > self.max_features:
                # If we have column importance, use it for feature selection
                if self.importances:
                    column_pairs = self._select_pairs_by_importance(column_pairs, self.max_features)
                else:
                    # Otherwise, select randomly
                    np.random.seed(self.random_state)
                    column_pairs = np.random.choice(column_pairs, self.max_features, replace=False)
            
            # Create interaction features
            for col1, col2 in column_pairs:
                new_col_name = f"{col1}_{interaction_type}_{col2}"
                try:
                    # Store the new column data rather than adding it directly to result
                    new_columns[new_col_name] = interact_func(X[col1].values, X[col2].values)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Error creating interaction {col1} {interaction_type} {col2}: {str(e)}")
        
        # Combine original DataFrame with all new columns at once
        if new_columns:
            new_df = pd.DataFrame(new_columns, index=X.index)
            if not result.empty:
                result = pd.concat([result, new_df], axis=1)
            else:
                result = new_df
        
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