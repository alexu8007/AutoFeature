"""
Mathematical Transformer

This module provides transformers for applying mathematical functions to features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from autofeature.feature_generation.base import BaseTransformer


class MathematicalTransformer(BaseTransformer):
    """Transformer for applying mathematical operations to features.
    
    This transformer applies mathematical functions like log, square, sqrt, etc.
    to numerical features.
    """
    
    SUPPORTED_OPERATIONS = {
        'square': ('square', lambda x: x ** 2),
        'cube': ('cube', lambda x: x ** 3),
        'sqrt': ('sqrt', lambda x: np.sqrt(np.abs(x))),
        'log': ('log', lambda x: np.log1p(np.abs(x))),
        'reciprocal': ('reciprocal', lambda x: 1 / (x + np.finfo(float).eps)),
        'sin': ('sin', np.sin),
        'cos': ('cos', np.cos),
        'tan': ('tan', np.tan),
        'abs': ('abs', np.abs),
        'negative': ('negative', lambda x: -x),
        'exp': ('exp', np.exp),
        'round': ('round', np.round),
        'sign': ('sign', np.sign),
        'sigmoid': ('sigmoid', lambda x: 1 / (1 + np.exp(-x))),
    }
    
    def __init__(self, operations: List[str] = None, custom_operations: Dict[str, Callable] = None, 
                 apply_to_all: bool = True, exclude_columns: List[str] = None, 
                 include_columns: List[str] = None):
        """Initialize the mathematical transformer.
        
        Args:
            operations: List of mathematical operations to apply
            custom_operations: Custom operations to apply, as {name: function}
            apply_to_all: Whether to apply to all numerical features
            exclude_columns: Columns to exclude from transformation
            include_columns: Only apply to these columns (if specified)
        """
        super().__init__()
        self.operations = operations or ['square', 'log', 'sqrt']
        self.custom_operations = custom_operations or {}
        self.apply_to_all = apply_to_all
        self.exclude_columns = exclude_columns or []
        self.include_columns = include_columns
        self.numerical_columns = []
        
        # Validate operations
        for op in self.operations:
            if op not in self.SUPPORTED_OPERATIONS and op not in self.custom_operations:
                raise ValueError(f"Operation '{op}' is not supported. "
                                f"Supported operations are: {list(self.SUPPORTED_OPERATIONS.keys())}. "
                                f"Alternatively, provide a custom operation.")
                
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MathematicalTransformer':
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
        """Transform the input data by applying mathematical operations.
        
        Args:
            X: Input features
            
        Returns:
            pd.DataFrame: Transformed data with new features
        """
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted yet. Call fit or fit_transform first.")
        
        self._validate_input(X)
        result = X.copy()
        
        for col in self.numerical_columns:
            if col not in X.columns:
                continue
                
            # Get the column data
            col_data = X[col].values
            
            # Apply each operation
            for op_name in self.operations:
                # Skip if column values not suitable (e.g., negative values for log)
                if not self._is_operation_applicable(op_name, col_data):
                    continue
                    
                # Get the operation function
                if op_name in self.SUPPORTED_OPERATIONS:
                    suffix, op_func = self.SUPPORTED_OPERATIONS[op_name]
                else:
                    suffix, op_func = op_name, self.custom_operations[op_name]
                
                # Generate the new feature name
                new_col_name = f"{col}_{suffix}"
                
                # Apply the operation
                try:
                    result[new_col_name] = op_func(col_data)
                    
                    # Register the feature
                    self._register_feature(new_col_name, {
                        'source_column': col,
                        'operation': op_name,
                        'type': 'mathematical'
                    })
                except Exception as e:
                    # Skip on error
                    continue
        
        return result
    
    def _is_operation_applicable(self, op_name: str, data: np.ndarray) -> bool:
        """Check if the operation is applicable to the data.
        
        Args:
            op_name: Name of the operation
            data: Column data
            
        Returns:
            bool: Whether the operation is applicable
        """
        # Skip operations that would produce NaN or infinite values
        if op_name == 'log' and np.any(data <= 0):
            return False
        if op_name == 'sqrt' and np.any(data < 0):
            return False
        if op_name == 'reciprocal' and np.any(data == 0):
            return False
            
        return True 