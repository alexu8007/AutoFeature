"""
Embedded Methods for Feature Selection

This module provides embedded methods for feature selection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from autofeature.feature_selection.base import BaseSelector


class EmbeddedSelector(BaseSelector):
    """Embedded feature selector.
    
    This selector uses models with built-in feature selection capabilities,
    such as Lasso, RandomForest, or other regularized models.
    """
    
    SUPPORTED_MODELS = {
        'regression': {
            'lasso': (Lasso, {'alpha': 0.1, 'random_state': 42}),
            'random_forest': (RandomForestRegressor, {'n_estimators': 100, 'random_state': 42})
        },
        'classification': {
            'logistic': (LogisticRegression, {
                'penalty': 'l1', 'C': 1.0, 'solver': 'liblinear', 'random_state': 42
            }),
            'random_forest': (RandomForestClassifier, {'n_estimators': 100, 'random_state': 42}),
            'linear_svc': (LinearSVC, {'C': 0.01, 'penalty': 'l1', 'dual': False, 'random_state': 42})
        }
    }
    
    def __init__(self, model_type: str = 'auto',
                 model_name: Optional[str] = None,
                 model: Optional[Any] = None,
                 threshold: Union[str, float] = 'mean',
                 max_features: Optional[int] = None,
                 prefit: bool = False,
                 importance_getter: Union[str, Callable] = 'auto',
                 verbose: int = 0):
        """Initialize the embedded selector.
        
        Args:
            model_type: Type of model to use ('regression', 'classification', 'auto')
            model_name: Name of predefined model to use
            model: Pre-configured model to use (if None, a default model is used)
            threshold: Feature selection threshold ('mean', 'median', or float)
            max_features: Maximum number of features to select
            prefit: Whether the provided model is already fitted
            importance_getter: Method for extracting feature importances
            verbose: Verbosity level
        """
        super().__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.model = model
        self.threshold = threshold
        self.max_features = max_features
        self.prefit = prefit
        self.importance_getter = importance_getter
        self.verbose = verbose
        self.fitted_model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EmbeddedSelector':
        """Fit the selector to the data.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            self: The fitted selector
        """
        self._validate_input(X)
        
        # Detect task type if auto
        if self.model_type == 'auto':
            is_regression = self._is_regression_task(y)
            self.model_type = 'regression' if is_regression else 'classification'
        
        # Setup model
        if self.model is None:
            model = self._get_default_model()
        else:
            model = self.model
        
        # Handle importance getter
        importance_getter = self._resolve_importance_getter(model)
        
        # Create selector
        selector = SelectFromModel(
            estimator=model,
            threshold=self.threshold,
            prefit=self.prefit,
            max_features=self.max_features,
            importance_getter=importance_getter
        )
        
        try:
            # Fit the selector
            selector.fit(X, y)
            
            # Get selected features
            feature_mask = selector.get_support()
            self.selected_features = [col for i, col in enumerate(X.columns) if feature_mask[i]]
            
            # Store the fitted model
            self.fitted_model = selector.estimator_
            
            # Get feature importances
            self._extract_feature_importances(X)
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error during embedded feature selection: {str(e)}")
                print("Falling back to default importance-based selection")
                
            # Fallback to direct model fitting and importance extraction
            try:
                # Fit the model directly
                if not self.prefit:
                    model.fit(X, y)
                self.fitted_model = model
                
                # Extract importances and select top features
                self._extract_feature_importances(X)
                
                # Select top features based on importance
                if self.max_features is not None:
                    k = min(self.max_features, len(X.columns))
                    sorted_features = sorted(
                        self.feature_importances_.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    self.selected_features = [f[0] for f in sorted_features[:k]]
                else:
                    # Use threshold on importances
                    if self.threshold == 'mean':
                        threshold = np.mean(list(self.feature_importances_.values()))
                    elif self.threshold == 'median':
                        threshold = np.median(list(self.feature_importances_.values()))
                    else:
                        threshold = self.threshold
                        
                    self.selected_features = [
                        col for col, imp in self.feature_importances_.items()
                        if imp >= threshold
                    ]
            except Exception as inner_e:
                if self.verbose > 0:
                    print(f"Fallback also failed: {str(inner_e)}")
                # Last resort: select all features
                self.selected_features = list(X.columns)
                self.feature_importances_ = {col: 1.0 / len(X.columns) for col in X.columns}
        
        self.is_fitted = True
        return self
    
    def _get_default_model(self) -> Any:
        """Get a default model based on model type and name.
        
        Returns:
            Any: Model instance
        """
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type '{self.model_type}' is not supported. "
                            f"Supported types are: {list(self.SUPPORTED_MODELS.keys())}.")
        
        models = self.SUPPORTED_MODELS[self.model_type]
        
        if self.model_name is None:
            # Default to random forest
            model_class, params = models['random_forest']
            return model_class(**params)
        
        if self.model_name not in models:
            raise ValueError(f"Model name '{self.model_name}' is not supported for {self.model_type}. "
                            f"Supported models are: {list(models.keys())}.")
        
        model_class, params = models[self.model_name]
        return model_class(**params)
    
    def _resolve_importance_getter(self, model: Any) -> Union[str, Callable]:
        """Resolve the importance getter based on the model.
        
        Args:
            model: Model instance
            
        Returns:
            Union[str, Callable]: Importance getter
        """
        if self.importance_getter != 'auto':
            return self.importance_getter
            
        # Try to infer the best importance getter for the model
        model_type = type(model).__name__.lower()
        
        if 'randomforest' in model_type or 'extratrees' in model_type:
            return 'feature_importances_'
        elif 'linear' in model_type or 'logistic' in model_type or 'lasso' in model_type or 'ridge' in model_type:
            return 'coef_'
        else:
            # Default
            return 'auto'
    
    def _extract_feature_importances(self, X: pd.DataFrame) -> None:
        """Extract feature importances from the fitted model.
        
        Args:
            X: Input features
        """
        if self.fitted_model is None:
            self.feature_importances_ = {col: 1.0 / len(X.columns) for col in X.columns}
            return
            
        try:
            # Get importances from model
            importances = self._get_importances_from_model(self.fitted_model)
            
            # Make sure importances is 1D
            if importances.ndim > 1:
                # For multi-class models, take the max importance across classes
                importances = np.abs(importances).max(axis=0)
                
            # Map importances to feature names
            self.feature_importances_ = dict(zip(X.columns, importances))
            
            # Normalize importances
            max_imp = max(self.feature_importances_.values()) if self.feature_importances_ else 1.0
            if max_imp > 0:
                self.feature_importances_ = {k: v/max_imp for k, v in self.feature_importances_.items()}
                
        except Exception as e:
            if self.verbose > 0:
                print(f"Error extracting feature importances: {str(e)}")
            # Fallback
            self.feature_importances_ = {col: 1.0 / len(X.columns) for col in X.columns}
    
    def _get_importances_from_model(self, model: Any) -> np.ndarray:
        """Get feature importances from the model.
        
        Args:
            model: Fitted model
            
        Returns:
            np.ndarray: Feature importances
        """
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
        elif hasattr(model, 'estimator') and hasattr(model.estimator, 'feature_importances_'):
            return model.estimator.feature_importances_
        elif hasattr(model, 'estimator') and hasattr(model.estimator, 'coef_'):
            return np.abs(model.estimator.coef_)
        else:
            raise AttributeError("Model does not have feature_importances_ or coef_ attributes")
    
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