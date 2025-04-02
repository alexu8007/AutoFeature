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
        """Transform the input data by creating aggregated features.
        
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
        
        # For each group by column
        for group_col in self.groupby_columns:
            if group_col not in X.columns:
                continue
                
            # If this column has too many unique values, skip it
            if X[group_col].nunique() > 100:  # Avoid creating too many features
                continue
                
            # Get group mapping for aggregated values
            agg_maps = {}
            
            # For each numerical column to aggregate
            for num_col in self.numerical_columns:
                if num_col not in X.columns or feature_count >= self.max_features:
                    continue
                    
                # For each aggregation type
                for agg_name in self.aggregation_types:
                    if feature_count >= self.max_features:
                        break
                        
                    # Get the aggregation function
                    if agg_name in self.SUPPORTED_AGGREGATIONS:
                        agg_func = self.SUPPORTED_AGGREGATIONS[agg_name]
                    else:
                        agg_func = self.custom_aggregations[agg_name]
                    
                    # Generate the new feature name
                    new_col_name = f"{num_col}_by_{group_col}_{agg_name}"
                    
                    try:
                        # Calculate aggregations
                        if group_col not in agg_maps:
                            agg_maps[group_col] = {}
                            
                        if agg_name not in agg_maps[group_col]:
                            agg_maps[group_col][agg_name] = {}
                            
                        # Calculate aggregation values for each group
                        for group_val, group_df in X.groupby(group_col):
                            if pd.isna(group_val):
                                continue  # Skip NaN group values
                                
                            try:
                                agg_maps[group_col][agg_name][group_val] = agg_func(group_df[num_col].values)
                            except Exception:
                                # Skip if aggregation fails for this group
                                continue
                        
                        # Create a mapping function to apply to the original column
                        def map_agg_value(x):
                            if pd.isna(x) or x not in agg_maps[group_col][agg_name]:
                                return np.nan
                            return agg_maps[group_col][agg_name][x]
                        
                        # Apply the mapping
                        result[new_col_name] = X[group_col].map(lambda x: map_agg_value(x))
                        
                        # Register the feature
                        self._register_feature(new_col_name, {
                            'group_column': group_col,
                            'agg_column': num_col,
                            'aggregation': agg_name,
                            'type': 'aggregation'
                        })
                        
                        feature_count += 1
                    except Exception as e:
                        # Skip on error
                        continue
        
        return result 