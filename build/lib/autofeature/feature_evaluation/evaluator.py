"""
Feature Evaluator

This module provides methods for evaluating the impact of features on model performance.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class FeatureEvaluator:
    """Evaluator for measuring the impact of features on model performance.
    
    This class provides methods for assessing feature importance and the 
    contribution of each feature to model performance.
    """
    
    SUPPORTED_METRICS = {
        'regression': {
            'r2': r2_score,
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error,
        },
        'classification': {
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': lambda y_true, y_pred: roc_auc_score(
                y_true, y_pred, multi_class='ovr', average='weighted'
            ) if len(np.unique(y_true)) > 2 else roc_auc_score(y_true, y_pred),
        }
    }
    
    def __init__(self, model_type: str = 'auto',
                 model: Optional[Any] = None,
                 cv: Union[int, Any] = 5,
                 metrics: Union[str, List[str]] = None,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 verbose: int = 0):
        """Initialize the feature evaluator.
        
        Args:
            model_type: Type of model to use ('regression', 'classification', 'auto')
            model: Pre-configured model to use (if None, a default model is used)
            cv: Cross-validation strategy
            metrics: Evaluation metrics
            n_jobs: Number of parallel jobs
            random_state: Random seed for reproducibility
            verbose: Verbosity level
        """
        self.model_type = model_type
        self.model = model
        self.cv = cv
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.feature_importances_ = {}
        self.feature_permutation_importances_ = {}
        self.feature_drop_importances_ = {}
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureEvaluator':
        """Fit the evaluator to the data.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            self: The fitted evaluator
        """
        # Detect task type if auto
        if self.model_type == 'auto':
            is_regression = self._is_regression_task(y)
            self.model_type = 'regression' if is_regression else 'classification'
        
        # Setup model and metrics
        model = self._setup_model()
        metrics = self._setup_metrics()
        
        # Setup cross-validation
        cv = self._setup_cv(y)
        
        try:
            # Calculate baseline performance
            self.baseline_scores_ = {}
            cv_results = cross_validate(
                model, X, y, cv=cv, scoring=metrics, n_jobs=self.n_jobs
            )
            
            for metric_name in metrics:
                # Convert from sklearn scoring format (e.g. 'test_r2') to our format
                score_key = f"test_{metric_name}"
                self.baseline_scores_[metric_name] = np.mean(cv_results[score_key])
            
            # Calculate feature importances from model
            self._calculate_model_feature_importances(X, y, model)
            
            # Calculate permutation importances
            self._calculate_permutation_importances(X, y, model, cv, metrics)
            
            # Calculate drop importances
            self._calculate_drop_importances(X, y, model, cv, metrics)
            
            self.is_fitted = True
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error during feature evaluation: {str(e)}")
        
        return self
    
    def evaluate_features(self, top_n: int = 10) -> pd.DataFrame:
        """Evaluate and rank features based on their importance.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            pd.DataFrame: DataFrame with feature evaluations
        """
        if not self.is_fitted:
            raise ValueError("Evaluator is not fitted yet. Call fit first.")
        
        results = []
        
        # Combine importances from different methods
        for feature in self.feature_importances_:
            result = {
                'feature': feature,
                'model_importance': self.feature_importances_.get(feature, 0.0),
            }
            
            # Add permutation importances for each metric
            for metric, importances in self.feature_permutation_importances_.items():
                if feature in importances:
                    result[f'permutation_{metric}'] = importances[feature]
            
            # Add drop importances for each metric
            for metric, importances in self.feature_drop_importances_.items():
                if feature in importances:
                    result[f'drop_{metric}'] = importances[feature]
            
            # Calculate aggregate importance
            result['aggregate_importance'] = np.mean([
                result['model_importance'],
                *[result[f'permutation_{m}'] for m in self.feature_permutation_importances_ 
                  if f'permutation_{m}' in result],
                *[result[f'drop_{m}'] for m in self.feature_drop_importances_ 
                  if f'drop_{m}' in result]
            ])
            
            results.append(result)
        
        # Create DataFrame and sort by aggregate importance
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('aggregate_importance', ascending=False)
        
        # Return top N features
        return results_df.head(top_n)
    
    def plot_feature_importances(self, top_n: int = 10, 
                                method: str = 'aggregate', 
                                metric: Optional[str] = None,
                                figsize: tuple = (10, 6)) -> None:
        """Plot feature importances.
        
        Args:
            top_n: Number of top features to show
            method: Importance method ('model', 'permutation', 'drop', 'aggregate')
            metric: Metric to use for permutation and drop importances
            figsize: Figure size
        """
        if not self.is_fitted:
            raise ValueError("Evaluator is not fitted yet. Call fit first.")
            
        plt.figure(figsize=figsize)
        
        if method == 'model':
            # Plot model importances
            importances = self.feature_importances_
            title = "Model Feature Importances"
        elif method == 'permutation':
            # Plot permutation importances
            if metric is None and self.feature_permutation_importances_:
                # Use first available metric
                metric = next(iter(self.feature_permutation_importances_))
                
            if metric not in self.feature_permutation_importances_:
                raise ValueError(f"Metric '{metric}' not found in permutation importances.")
                
            importances = self.feature_permutation_importances_[metric]
            title = f"Permutation Feature Importances ({metric})"
        elif method == 'drop':
            # Plot drop importances
            if metric is None and self.feature_drop_importances_:
                # Use first available metric
                metric = next(iter(self.feature_drop_importances_))
                
            if metric not in self.feature_drop_importances_:
                raise ValueError(f"Metric '{metric}' not found in drop importances.")
                
            importances = self.feature_drop_importances_[metric]
            title = f"Drop Feature Importances ({metric})"
        elif method == 'aggregate':
            # Calculate aggregate importances
            importances = {}
            for feature in self.feature_importances_:
                values = [self.feature_importances_.get(feature, 0.0)]
                
                # Add permutation importances
                for m_importances in self.feature_permutation_importances_.values():
                    if feature in m_importances:
                        values.append(m_importances[feature])
                
                # Add drop importances
                for m_importances in self.feature_drop_importances_.values():
                    if feature in m_importances:
                        values.append(m_importances[feature])
                
                importances[feature] = np.mean(values)
                
            title = "Aggregate Feature Importances"
        else:
            raise ValueError(f"Method '{method}' is not supported. "
                            f"Supported methods are: 'model', 'permutation', 'drop', 'aggregate'.")
        
        # Sort importances
        sorted_importances = sorted(
            importances.items(), key=lambda x: x[1], reverse=True
        )
        
        # Get top N features
        top_features = sorted_importances[:top_n]
        
        # Plot
        features, values = zip(*top_features)
        
        # Create horizontal bar plot
        plt.barh(range(len(features)), values, align='center')
        plt.yticks(range(len(features)), features)
        plt.gca().invert_yaxis()  # Highest values at the top
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        
        return plt.gcf()
    
    def feature_correlation_matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate the correlation matrix between features.
        
        Args:
            X: Input features
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Get numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns
        
        # Calculate correlation matrix
        corr_matrix = X[numeric_cols].corr()
        
        return corr_matrix
    
    def plot_correlation_matrix(self, X: pd.DataFrame, 
                               figsize: tuple = (12, 10),
                               cmap: str = 'coolwarm',
                               mask_upper: bool = True) -> None:
        """Plot the correlation matrix between features.
        
        Args:
            X: Input features
            figsize: Figure size
            cmap: Colormap
            mask_upper: Whether to mask the upper triangle
        """
        # Calculate correlation matrix
        corr_matrix = self.feature_correlation_matrix(X)
        
        # Create mask for upper triangle
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create plot
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            mask=mask,
            cmap=cmap, 
            vmin=-1, 
            vmax=1, 
            center=0,
            linewidths=.5, 
            fmt=".2f"
        )
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        return plt.gcf()
    
    def _setup_model(self) -> Any:
        """Set up the model for feature evaluation.
        
        Returns:
            Any: Model instance
        """
        if self.model is not None:
            return self.model
        
        # Create default model based on task type
        if self.model_type == 'regression':
            return RandomForestRegressor(n_estimators=100, n_jobs=self.n_jobs, random_state=self.random_state)
        else:
            return RandomForestClassifier(n_estimators=100, n_jobs=self.n_jobs, random_state=self.random_state)
    
    def _setup_metrics(self) -> Union[str, List[str]]:
        """Set up evaluation metrics.
        
        Returns:
            Union[str, List[str]]: Evaluation metrics
        """
        if self.metrics is not None:
            # Convert single metric to list
            if isinstance(self.metrics, str):
                return [self.metrics]
            return self.metrics
        
        # Default metrics based on task type
        if self.model_type == 'regression':
            return ['r2', 'neg_mean_squared_error']
        else:
            return ['accuracy', 'f1_weighted']
    
    def _setup_cv(self, y: pd.Series) -> Any:
        """Set up cross-validation strategy.
        
        Args:
            y: Target variable
            
        Returns:
            Any: Cross-validation strategy
        """
        if isinstance(self.cv, int):
            # Create cross-validation based on task type
            if self.model_type == 'regression':
                return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            else:
                return StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # Return custom CV strategy
        return self.cv
    
    def _calculate_model_feature_importances(self, X: pd.DataFrame, y: pd.Series, model: Any) -> None:
        """Calculate feature importances from model.
        
        Args:
            X: Input features
            y: Target variable
            model: Model to use
        """
        try:
            # Fit model
            model.fit(X, y)
            
            # Get feature importances based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    # For multi-class models, take the mean across classes
                    importances = np.mean(importances, axis=0)
            else:
                # Fallback
                importances = np.ones(X.shape[1]) / X.shape[1]
            
            # Normalize importances
            if max(importances) > 0:
                importances = importances / max(importances)
                
            # Map to feature names
            self.feature_importances_ = dict(zip(X.columns, importances))
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Error calculating model feature importances: {str(e)}")
            # Fallback
            self.feature_importances_ = {col: 1.0 / X.shape[1] for col in X.columns}
    
    def _calculate_permutation_importances(self, X: pd.DataFrame, y: pd.Series, 
                                          model: Any, cv: Any, metrics: List[str]) -> None:
        """Calculate permutation importances.
        
        Args:
            X: Input features
            y: Target variable
            model: Model to use
            cv: Cross-validation strategy
            metrics: Evaluation metrics
        """
        self.feature_permutation_importances_ = {}
        
        try:
            from sklearn.inspection import permutation_importance
            
            # Calculate permutation importance for each metric
            for metric in metrics:
                # Skip metrics that are not supported by permutation_importance
                if metric.startswith('neg_'):
                    scoring = metric
                else:
                    # Check if metric exists in sklearn's SCORERS dictionary
                    try:
                        from sklearn.metrics import get_scorer
                        get_scorer(metric)
                        scoring = metric
                    except:
                        # Skip this metric
                        continue
                
                # Calculate permutation importance
                result = permutation_importance(
                    model, X, y, scoring=scoring, n_repeats=5, 
                    n_jobs=self.n_jobs, random_state=self.random_state
                )
                
                # Get mean importance scores
                importances = result.importances_mean
                
                # Normalize importances
                if max(importances) > 0:
                    importances = importances / max(importances)
                    
                # Map to feature names
                self.feature_permutation_importances_[metric] = dict(zip(X.columns, importances))
                
        except (ImportError, Exception) as e:
            if self.verbose > 0:
                print(f"Error calculating permutation importances: {str(e)}")
    
    def _calculate_drop_importances(self, X: pd.DataFrame, y: pd.Series, 
                                   model: Any, cv: Any, metrics: List[str]) -> None:
        """Calculate drop importances by measuring performance change when features are removed.
        
        Args:
            X: Input features
            y: Target variable
            model: Model to use
            cv: Cross-validation strategy
            metrics: Evaluation metrics
        """
        self.feature_drop_importances_ = defaultdict(dict)
        
        try:
            # Get baseline scores
            baseline_scores = self.baseline_scores_
            
            # Calculate importance by dropping each feature
            for feature in X.columns:
                # Create data without this feature
                X_drop = X.drop(columns=[feature])
                
                # Calculate scores without this feature
                cv_results = cross_validate(
                    model, X_drop, y, cv=cv, scoring=metrics, n_jobs=self.n_jobs
                )
                
                # Calculate importance as performance drop
                for metric in metrics:
                    score_key = f"test_{metric}"
                    if score_key in cv_results:
                        drop_score = np.mean(cv_results[score_key])
                        baseline = baseline_scores.get(metric, 0)
                        
                        # Importance is performance drop (or gain for negative metrics)
                        if metric.startswith('neg_'):
                            # For negative metrics, lower is better
                            importance = (drop_score - baseline) / abs(baseline) if baseline != 0 else 0
                        else:
                            # For positive metrics, higher is better
                            importance = (baseline - drop_score) / abs(baseline) if baseline != 0 else 0
                        
                        self.feature_drop_importances_[metric][feature] = max(0, importance)
            
            # Normalize importances for each metric
            for metric in self.feature_drop_importances_:
                importances = list(self.feature_drop_importances_[metric].values())
                if importances and max(importances) > 0:
                    # Normalize
                    self.feature_drop_importances_[metric] = {
                        k: v / max(importances) 
                        for k, v in self.feature_drop_importances_[metric].items()
                    }
                    
        except Exception as e:
            if self.verbose > 0:
                print(f"Error calculating drop importances: {str(e)}")
    
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