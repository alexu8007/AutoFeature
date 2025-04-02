"""
Wrapper Methods for Feature Selection

This module provides wrapper-based methods for feature selection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score, accuracy_score
from autofeature.feature_selection.base import BaseSelector


class WrapperSelector(BaseSelector):
    """Wrapper-based feature selector.
    
    This selector uses model performance to select features through
    forward selection, backward elimination, or recursive feature elimination.
    """
    
    SUPPORTED_METHODS = ['forward', 'backward', 'recursive']
    
    SUPPORTED_MODELS = {
        'regression': {
            'linear': LinearRegression,
            'random_forest': RandomForestRegressor
        },
        'classification': {
            'logistic': LogisticRegression,
            'random_forest': RandomForestClassifier
        }
    }
    
    def __init__(self, method: str = 'forward', 
                 model_type: str = 'auto',
                 model: Optional[Any] = None,
                 cv: int = 5,
                 scoring: Union[str, Callable] = None,
                 k: Union[int, float] = 10,
                 min_features: int = 1,
                 max_features: Optional[int] = None,
                 n_jobs: int = -1,
                 verbose: int = 0):
        """Initialize the wrapper selector.
        
        Args:
            method: Method for feature selection ('forward', 'backward', 'recursive')
            model_type: Type of model to use ('regression', 'classification', 'auto')
            model: Pre-configured model to use (if None, a default model is used)
            cv: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            k: Number of features to select (if int) or fraction (if float < 1.0)
            min_features: Minimum number of features to select
            max_features: Maximum number of features to consider
            n_jobs: Number of parallel jobs for cross-validation
            verbose: Verbosity level
        """
        super().__init__()
        self.method = method
        self.model_type = model_type
        self.model = model
        self.cv = cv
        self.scoring = scoring
        self.k = k
        self.min_features = min_features
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Validate method
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method '{method}' is not supported. "
                            f"Supported methods are: {self.SUPPORTED_METHODS}.")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'WrapperSelector':
        """Fit the selector to the data by evaluating feature subsets.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            self: The fitted selector
        """
        self._validate_input(X)
        
        # Initialize feature importance dict
        self.feature_importances_ = {col: 0.0 for col in X.columns}
        
        # Detect task type if auto
        if self.model_type == 'auto':
            is_regression = self._is_regression_task(y)
            self.model_type = 'regression' if is_regression else 'classification'
        
        # Setup model and scoring
        model = self._setup_model()
        scoring = self._setup_scoring()
        
        # Get the max number of features to consider
        n_features = X.shape[1]
        if self.max_features is None:
            max_features = n_features
        else:
            max_features = min(self.max_features, n_features)
        
        # Get target number of features
        if isinstance(self.k, float) and self.k < 1.0:
            k = max(self.min_features, int(self.k * n_features))
        else:
            k = min(self.k, n_features)
        
        # Apply the selected method
        if self.method == 'forward':
            self.selected_features, self.feature_importances_ = self._forward_selection(
                X, y, model, scoring, k, max_features
            )
        elif self.method == 'backward':
            self.selected_features, self.feature_importances_ = self._backward_elimination(
                X, y, model, scoring, k, max_features
            )
        elif self.method == 'recursive':
            self.selected_features, self.feature_importances_ = self._recursive_feature_elimination(
                X, y, model, scoring, k, max_features
            )
        
        self.is_fitted = True
        return self
    
    def _setup_model(self) -> Any:
        """Set up the model for feature selection.
        
        Returns:
            Any: Model instance
        """
        if self.model is not None:
            return self.model
        
        # Create default model based on task type
        if self.model_type == 'regression':
            return RandomForestRegressor(n_estimators=100, n_jobs=self.n_jobs, random_state=42)
        else:
            return RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs, random_state=42)
    
    def _setup_scoring(self) -> Union[str, Callable]:
        """Set up scoring metric for cross-validation.
        
        Returns:
            Union[str, Callable]: Scoring metric
        """
        if self.scoring is not None:
            return self.scoring
        
        # Default scoring based on task type
        if self.model_type == 'regression':
            return 'r2'
        else:
            return 'accuracy'
    
    def _evaluate_feature_set(self, X: pd.DataFrame, y: pd.Series, 
                             features: List[str], model: Any, 
                             scoring: Union[str, Callable]) -> float:
        """Evaluate a feature set using cross-validation.
        
        Args:
            X: Input features
            y: Target variable
            features: Features to evaluate
            model: Model to use
            scoring: Scoring metric
            
        Returns:
            float: Mean cross-validation score
        """
        if not features:
            return 0.0
            
        # Create a copy of the model to avoid fitting the same model multiple times
        model_copy = clone_model(model)
        
        try:
            # Evaluate using cross-validation
            scores = cross_val_score(
                model_copy, X[features], y, 
                cv=self.cv, scoring=scoring, n_jobs=self.n_jobs
            )
            return np.mean(scores)
        except Exception:
            # Return very low score on error
            return -np.inf
    
    def _forward_selection(self, X: pd.DataFrame, y: pd.Series, 
                          model: Any, scoring: Union[str, Callable], 
                          k: int, max_features: int) -> tuple:
        """Perform forward selection.
        
        Args:
            X: Input features
            y: Target variable
            model: Model to use
            scoring: Scoring metric
            k: Number of features to select
            max_features: Maximum number of features to consider
            
        Returns:
            tuple: Selected features and feature importances
        """
        # Start with empty feature set
        selected = []
        current_score = 0.0
        remaining = list(X.columns)
        importances = {col: 0.0 for col in X.columns}
        
        for i in range(min(k, max_features)):
            best_score = -np.inf
            best_feature = None
            
            if self.verbose > 0:
                print(f"Forward selection step {i+1}/{k}")
                
            # Try adding each remaining feature
            for feature in remaining:
                score = self._evaluate_feature_set(
                    X, y, selected + [feature], model, scoring
                )
                
                # Update best score and feature
                if score > best_score:
                    best_score = score
                    best_feature = feature
            
            # Stop if no improvement
            if best_score <= current_score or best_feature is None:
                break
                
            # Add best feature
            if self.verbose > 0:
                print(f"  Selected {best_feature} (score: {best_score:.4f})")
                
            selected.append(best_feature)
            remaining.remove(best_feature)
            current_score = best_score
            
            # Record feature importance (based on order of selection)
            importances[best_feature] = len(selected)
            
        # Normalize importances
        max_imp = max(importances.values()) if importances else 1.0
        importances = {k: v/max_imp for k, v in importances.items()}
            
        return selected, importances
    
    def _backward_elimination(self, X: pd.DataFrame, y: pd.Series, 
                             model: Any, scoring: Union[str, Callable], 
                             k: int, max_features: int) -> tuple:
        """Perform backward elimination.
        
        Args:
            X: Input features
            y: Target variable
            model: Model to use
            scoring: Scoring metric
            k: Number of features to select
            max_features: Maximum number of features to consider
            
        Returns:
            tuple: Selected features and feature importances
        """
        # Start with all features
        all_features = list(X.columns)[:max_features]
        selected = all_features.copy()
        
        # Evaluate initial feature set
        current_score = self._evaluate_feature_set(
            X, y, selected, model, scoring
        )
        
        importances = {col: 1.0 for col in all_features}
        
        # Remove features until we reach k
        while len(selected) > k:
            worst_score = -np.inf
            worst_feature = None
            
            if self.verbose > 0:
                print(f"Backward elimination step {len(all_features) - len(selected) + 1}/{len(all_features) - k}")
                
            # Try removing each feature
            for feature in selected:
                temp_selected = [f for f in selected if f != feature]
                score = self._evaluate_feature_set(
                    X, y, temp_selected, model, scoring
                )
                
                # Update worst feature (the one that when removed gives highest score)
                if score > worst_score:
                    worst_score = score
                    worst_feature = feature
            
            # If removing any feature doesn't improve score, break
            if worst_score <= current_score:
                break
                
            # Remove worst feature
            if self.verbose > 0 and worst_feature is not None:
                print(f"  Removed {worst_feature} (score: {worst_score:.4f})")
                
            if worst_feature is not None:
                selected.remove(worst_feature)
                importances[worst_feature] = 0.0
                current_score = worst_score
            else:
                break
                
        # Update importances based on presence in final set
        elimination_order = len(all_features) - len(selected)
        elimination_step = 0
        for feat in all_features:
            if feat not in selected:
                elimination_step += 1
                importances[feat] = 1.0 - (elimination_step / elimination_order) if elimination_order > 0 else 0.0
            
        return selected, importances
    
    def _recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                      model: Any, scoring: Union[str, Callable], 
                                      k: int, max_features: int) -> tuple:
        """Perform recursive feature elimination.
        
        Args:
            X: Input features
            y: Target variable
            model: Model to use
            scoring: Scoring metric
            k: Number of features to select
            max_features: Maximum number of features to consider
            
        Returns:
            tuple: Selected features and feature importances
        """
        try:
            from sklearn.feature_selection import RFECV
            
            # Create RFE model
            rfe = RFECV(
                estimator=model,
                step=1,
                min_features_to_select=k,
                cv=self.cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=self.verbose
            )
            
            # Select features (limit to max_features)
            features = list(X.columns)[:max_features]
            rfe.fit(X[features], y)
            
            # Get selected features
            selected = [features[i] for i, selected in enumerate(rfe.support_) if selected]
            
            # Get feature importances
            importances = {}
            for i, feature in enumerate(features):
                if rfe.support_[i]:
                    importances[feature] = rfe.ranking_[i]
                else:
                    importances[feature] = 0.0
                    
            # Normalize importances (higher values for more important features)
            max_rank = max(rfe.ranking_)
            importances = {k: (max_rank - v + 1) / max_rank for k, v in importances.items()}
            
            return selected, importances
            
        except ImportError:
            # Fallback to forward selection if scikit-learn RFECV is not available
            return self._forward_selection(X, y, model, scoring, k, max_features)
    
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


def clone_model(model):
    """Clone a scikit-learn model.
    
    Args:
        model: Model to clone
        
    Returns:
        Any: Cloned model
    """
    try:
        from sklearn.base import clone
        return clone(model)
    except ImportError:
        # Simple fallback - will not work for all models
        try:
            return model.__class__(**model.get_params())
        except:
            # Last resort - return the original model
            return model 