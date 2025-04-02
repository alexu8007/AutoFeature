"""
Time-Based Transformer

This module provides transformers for extracting features from datetime columns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from autofeature.feature_generation.base import BaseTransformer


class TimeBasedTransformer(BaseTransformer):
    """Transformer for extracting features from datetime columns.
    
    This transformer creates new features by extracting components from datetime
    columns, such as year, month, day, hour, etc., as well as creating cyclical
    features for periodic time components.
    """
    
    DATETIME_COMPONENTS = {
        'year': lambda x: x.dt.year,
        'month': lambda x: x.dt.month,
        'day': lambda x: x.dt.day,
        'day_of_week': lambda x: x.dt.dayofweek,
        'day_of_year': lambda x: x.dt.dayofyear,
        'quarter': lambda x: x.dt.quarter,
        'hour': lambda x: x.dt.hour,
        'minute': lambda x: x.dt.minute,
        'second': lambda x: x.dt.second,
        'is_weekend': lambda x: x.dt.dayofweek.isin([5, 6]).astype(int),
        'is_month_start': lambda x: x.dt.is_month_start.astype(int),
        'is_month_end': lambda x: x.dt.is_month_end.astype(int),
        'is_quarter_start': lambda x: x.dt.is_quarter_start.astype(int),
        'is_quarter_end': lambda x: x.dt.is_quarter_end.astype(int),
        'is_year_start': lambda x: x.dt.is_year_start.astype(int),
        'is_year_end': lambda x: x.dt.is_year_end.astype(int),
    }
    
    CYCLICAL_COMPONENTS = {
        'month': 12,
        'day': 31,
        'day_of_week': 7,
        'hour': 24,
        'minute': 60,
        'second': 60,
        'quarter': 4,
    }
    
    def __init__(self, components: List[str] = None, cyclical: bool = True,
                 datetime_format: str = None, datetime_columns: List[str] = None,
                 include_time_delta: bool = True, reference_date: Union[str, datetime] = None):
        """Initialize the time-based transformer.
        
        Args:
            components: Time components to extract
            cyclical: Whether to create cyclical features
            datetime_format: Format string for parsing datetime columns
            datetime_columns: Datetime columns to transform (if None, auto-detect)
            include_time_delta: Whether to include time delta features
            reference_date: Reference date for time delta calculations
        """
        super().__init__()
        self.components = components or ['year', 'month', 'day', 'day_of_week', 'hour']
        self.cyclical = cyclical
        self.datetime_format = datetime_format
        self.datetime_columns = datetime_columns
        self.include_time_delta = include_time_delta
        self.reference_date = reference_date
        self.detected_datetime_columns = []
        
        # Validate components
        for component in self.components:
            if component not in self.DATETIME_COMPONENTS:
                raise ValueError(f"Component '{component}' is not supported. "
                                f"Supported components are: {list(self.DATETIME_COMPONENTS.keys())}.")
                
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TimeBasedTransformer':
        """Fit the transformer to the data by identifying datetime columns.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            self: The fitted transformer
        """
        self._validate_input(X)
        
        # If datetime columns are provided, use those
        if self.datetime_columns:
            self.detected_datetime_columns = [col for col in self.datetime_columns if col in X.columns]
        else:
            # Otherwise, detect datetime columns
            self.detected_datetime_columns = []
            for col in X.columns:
                try:
                    # Check if column is already datetime
                    if pd.api.types.is_datetime64_any_dtype(X[col]):
                        self.detected_datetime_columns.append(col)
                    # Try to convert to datetime
                    elif X[col].dtype.kind in 'OSU':  # Object, string, or unicode
                        pd.to_datetime(X[col], format=self.datetime_format, errors='raise')
                        self.detected_datetime_columns.append(col)
                except (ValueError, TypeError):
                    continue
        
        # Prepare reference date for time delta features
        if self.include_time_delta and self.reference_date is None:
            try:
                # Use the minimum date as reference if not provided
                min_dates = []
                for col in self.detected_datetime_columns:
                    if pd.api.types.is_datetime64_any_dtype(X[col]):
                        min_dates.append(X[col].min())
                    else:
                        min_dates.append(pd.to_datetime(X[col], format=self.datetime_format).min())
                
                if min_dates:
                    self.reference_date = min(min_dates)
            except Exception:
                # If reference date cannot be determined, disable time delta
                self.include_time_delta = False
                
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data by extracting time-based features.
        
        Args:
            X: Input features
            
        Returns:
            pd.DataFrame: Transformed data with new features
        """
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted yet. Call fit or fit_transform first.")
        
        self._validate_input(X)
        result = X.copy()
        
        for col in self.detected_datetime_columns:
            if col not in X.columns:
                continue
                
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(X[col]):
                try:
                    dt_series = pd.to_datetime(X[col], format=self.datetime_format)
                except Exception:
                    continue
            else:
                dt_series = X[col]
                
            # Extract datetime components
            for component in self.components:
                if component not in self.DATETIME_COMPONENTS:
                    continue
                    
                try:
                    # Generate the new feature
                    new_col_name = f"{col}_{component}"
                    result[new_col_name] = self.DATETIME_COMPONENTS[component](dt_series)
                    
                    # Register the feature
                    self._register_feature(new_col_name, {
                        'source_column': col,
                        'component': component,
                        'type': 'time_component'
                    })
                    
                    # Create cyclical features for applicable components
                    if self.cyclical and component in self.CYCLICAL_COMPONENTS:
                        max_val = self.CYCLICAL_COMPONENTS[component]
                        
                        # Sin component
                        sin_col_name = f"{col}_{component}_sin"
                        result[sin_col_name] = np.sin(2 * np.pi * result[new_col_name] / max_val)
                        self._register_feature(sin_col_name, {
                            'source_column': col,
                            'component': component,
                            'cyclical': 'sin',
                            'type': 'time_cyclical'
                        })
                        
                        # Cos component
                        cos_col_name = f"{col}_{component}_cos"
                        result[cos_col_name] = np.cos(2 * np.pi * result[new_col_name] / max_val)
                        self._register_feature(cos_col_name, {
                            'source_column': col,
                            'component': component,
                            'cyclical': 'cos',
                            'type': 'time_cyclical'
                        })
                except Exception:
                    continue
            
            # Add time delta features if enabled
            if self.include_time_delta and self.reference_date is not None:
                try:
                    # Calculate time delta in days
                    delta_days_col = f"{col}_days_from_ref"
                    delta = (dt_series - pd.to_datetime(self.reference_date)).dt.total_seconds() / (24 * 3600)
                    result[delta_days_col] = delta
                    
                    self._register_feature(delta_days_col, {
                        'source_column': col,
                        'reference_date': str(self.reference_date),
                        'type': 'time_delta'
                    })
                    
                    # Calculate time delta in weeks
                    delta_weeks_col = f"{col}_weeks_from_ref"
                    result[delta_weeks_col] = delta / 7
                    
                    self._register_feature(delta_weeks_col, {
                        'source_column': col,
                        'reference_date': str(self.reference_date),
                        'type': 'time_delta'
                    })
                    
                    # Calculate time delta in months (approximate)
                    delta_months_col = f"{col}_months_from_ref"
                    result[delta_months_col] = delta / 30.44  # Average days in a month
                    
                    self._register_feature(delta_months_col, {
                        'source_column': col,
                        'reference_date': str(self.reference_date),
                        'type': 'time_delta'
                    })
                except Exception:
                    continue
        
        return result 