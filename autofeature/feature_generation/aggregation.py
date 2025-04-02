"""
Aggregation Transformer

This module provides transformers for creating aggregated features based on group-by operations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Union
from autofeature.feature_generation.base import BaseTransformer


class AggregationTransformer(BaseTransformer):
    """Transformer for creating aggregated features.
    
    This transformer creates new features by applying aggregation functions
    to grouped data, such as mean, sum, std, etc. of a numerical column
    grouped by a categorical column.
    """
    
    SUPPORTED_AGGREGATIONS = {
        'mean': np.mean,
        'median': np.median,
        'std': np.std,
        'var': np.var,
        'min': np.min,
        'max': np.max,
        'sum': np.sum,
        'count': len,
        'nunique': lambda x: len(set(x)),
        'first': lambda x: x.iloc[0] if isinstance(x, pd.Series) else x[0],
        'last': lambda x: x.iloc[-1] if isinstance(x, pd.Series) else x[-1],
        'range': lambda x: np.max(x) - np.min(x),
        'iqr': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        'skew': lambda x: pd.Series(x).skew(),
        'kurtosis': lambda x: pd.Series(x).kurtosis(),
    }
    
    def __init__(self, aggregation_types: List[str] = None, custom_aggregations: Dict[str, Callable] = None,
                 groupby_columns: List[str] = None, agg_columns: List[str] = None,
                 max_features: int = 100):
        """Initialize the aggregation transformer.
        
        Args:
            aggregation_types: Types of aggregations to create
            custom_aggregations: Custom aggregation functions to apply
            groupby_columns: Columns to group by
            agg_columns: Columns to aggregate
            max_features: Maximum number of features to generate
        """
        super().__init__()
        self.aggregation_types = aggregation_types or ['mean', 'median', 'std', 'min', 'max']
        self.custom_aggregations = custom_aggregations or {}
        self.groupby_columns = groupby_columns
        self.agg_columns = agg_columns
        self.max_features = max_features
        self.numerical_columns = []
        self.categorical_columns = []
        
        # Validate aggregation types
        for agg in self.aggregation_types:
            if agg not in self.SUPPORTED_AGGREGATIONS and agg not in self.custom_aggregations:
                raise ValueError(f"Aggregation '{agg}' is not supported. "
                                f"Supported aggregations are: {list(self.SUPPORTED_AGGREGATIONS.keys())}. "
                                f"Alternatively, provide a custom aggregation.")
                
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AggregationTransformer':
        """Fit the transformer to the data by identifying numerical and categorical columns.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            self: The fitted transformer
        """
        self._validate_input(X)
        
        # Find numerical and categorical columns
        self.numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Filter aggregation columns if specified
        if self.agg_columns:
            self.numerical_columns = [col for col in self.numerical_columns if col in self.agg_columns]
            
        # If no groupby columns are specified, use all categorical columns
        if not self.groupby_columns:
            self.groupby_columns = self.categorical_columns
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by creating aggregation features.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input data to transform
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with aggregation features added
        """
        self._check_fitted()
        X = check_dataframe(X)
        
        # Create a copy of the input dataframe to avoid modifying it
        result = X.copy() if self.include_orig else pd.DataFrame(index=X.index)
        
        # Dictionary to collect all new columns
        new_columns = {}
        
        for group_col in self.groupby_columns:
            if group_col not in X.columns:
                if self.verbose > 0:
                    print(f"Warning: Groupby column {group_col} not found in the data")
                continue
            
            for agg_col in self.agg_columns:
                if agg_col not in X.columns:
                    if self.verbose > 0:
                        print(f"Warning: Aggregation column {agg_col} not found in the data")
                    continue
                
                # Skip if the aggregation column isn't numeric
                if not pd.api.types.is_numeric_dtype(X[agg_col]):
                    if self.verbose > 0:
                        print(f"Warning: Aggregation column {agg_col} is not numeric")
                    continue
                
                # Calculate aggregations for this group-agg column pair
                for agg_func in self.aggregation_types:
                    try:
                        # Calculate the aggregation
                        grouped = X.groupby(group_col)[agg_col].agg(agg_func)
                        
                        # Create column name
                        new_col_name = f"agg_{group_col}_{agg_col}_{agg_func}"
                        
                        # Create a mapping function to get aggregated values
                        map_agg_value = lambda x: grouped.get(x, np.nan)
                        
                        # Store the new column data instead of adding it directly
                        new_columns[new_col_name] = X[group_col].map(lambda x: map_agg_value(x))
                    
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Error creating aggregation for {group_col}, {agg_col}, {agg_func}: {str(e)}")
        
        # Combine original DataFrame with all new columns at once
        if new_columns:
            new_df = pd.DataFrame(new_columns, index=X.index)
            if not result.empty:
                result = pd.concat([result, new_df], axis=1)
            else:
                result = new_df
        
        return result 