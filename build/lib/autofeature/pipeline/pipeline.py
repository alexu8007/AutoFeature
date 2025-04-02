"""
Feature Pipeline

This module provides a pipeline for automated feature engineering.
"""

import pandas as pd
import numpy as np
import time
import json
import os
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, Type
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from autofeature.feature_generation.base import BaseTransformer
from autofeature.feature_selection.base import BaseSelector
from autofeature.feature_selection.filter_methods import FilterSelector
from autofeature.feature_selection.genetic_algorithm import GeneticSelector
from autofeature.feature_evaluation.evaluator import FeatureEvaluator


class FeaturePipeline:
    """Pipeline for automated feature engineering.
    
    This pipeline integrates feature generation, selection, and evaluation
    into a single workflow.
    """
    
    def __init__(self, generation_steps: Optional[List[BaseTransformer]] = None,
                 selection_method: Optional[BaseSelector] = None,
                 target_metric: Optional[str] = None,
                 max_features: Optional[int] = None,
                 n_jobs: int = -1,
                 random_state: Optional[int] = None,
                 verbose: int = 0,
                 preprocess: bool = True,
                 evaluate: bool = True):
        """Initialize the feature pipeline.
        
        Args:
            generation_steps: List of feature generator transformers
            selection_method: Feature selection method
            target_metric: Target metric for evaluation and selection
            max_features: Maximum number of features to select
            n_jobs: Number of parallel jobs
            random_state: Random seed for reproducibility
            verbose: Verbosity level
            preprocess: Whether to apply preprocessing steps (imputation, scaling)
            evaluate: Whether to evaluate features
        """
        self.generation_steps = generation_steps or []
        
        # Set up feature selection
        if selection_method is None:
            self.selection_method = GeneticSelector(
                n_jobs=n_jobs, random_state=random_state, 
                verbose=verbose, n_generations=5
            )
        else:
            self.selection_method = selection_method
            
        self.target_metric = target_metric
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.preprocess = preprocess
        self.evaluate = evaluate
        
        # Initialize other attributes
        self.is_fitted = False
        self.original_columns = []
        self.generated_columns = []
        self.selected_columns = []
        self.feature_importances_ = {}
        self.runtime_stats_ = {}
        self.evaluator = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeaturePipeline':
        """Fit the pipeline to the data.
        
        This will:
        1. Generate new features
        2. Select the best features
        3. Evaluate feature importance
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            self: The fitted pipeline
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array")
            
        # Convert y to pandas Series if it's a numpy array
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        # Store original columns
        self.original_columns = list(X.columns)
        
        # Initialize runtime stats
        self.runtime_stats_ = {
            'start_time': time.time(),
            'generation_time': 0,
            'selection_time': 0,
            'evaluation_time': 0,
            'total_time': 0,
            'original_features': len(self.original_columns),
            'generated_features': 0,
            'selected_features': 0
        }
        
        # Step 1: Preprocess data if enabled
        preprocess_start = time.time()
        if self.preprocess:
            X_processed = self._preprocess_data(X)
        else:
            X_processed = X.copy()
        self.runtime_stats_['preprocess_time'] = time.time() - preprocess_start
        
        # Step 2: Generate features
        generation_start = time.time()
        X_generated = self._generate_features(X_processed)
        self.runtime_stats_['generation_time'] = time.time() - generation_start
        
        # Store generated columns
        self.generated_columns = [col for col in X_generated.columns if col not in self.original_columns]
        self.runtime_stats_['generated_features'] = len(self.generated_columns)
        
        if self.verbose > 0:
            print(f"Generated {len(self.generated_columns)} new features. "
                 f"Total features: {X_generated.shape[1]}")
        
        # Step 3: Select features
        selection_start = time.time()
        X_selected, selected_columns = self._select_features(X_generated, y)
        self.runtime_stats_['selection_time'] = time.time() - selection_start
        
        # Store selected columns
        self.selected_columns = selected_columns
        self.runtime_stats_['selected_features'] = len(self.selected_columns)
        
        if self.verbose > 0:
            orig_selected = [col for col in self.selected_columns if col in self.original_columns]
            gen_selected = [col for col in self.selected_columns if col in self.generated_columns]
            print(f"Selected {len(self.selected_columns)} features: "
                 f"{len(orig_selected)} original, {len(gen_selected)} generated")
        
        # Step 4: Evaluate features
        if self.evaluate:
            evaluation_start = time.time()
            self._evaluate_features(X_generated, y)
            self.runtime_stats_['evaluation_time'] = time.time() - evaluation_start
        
        # Update total runtime
        self.runtime_stats_['total_time'] = time.time() - self.runtime_stats_['start_time']
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data by applying the fitted pipeline.
        
        Args:
            X: Input features
            
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet. Call fit or fit_transform first.")
            
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        # Step 1: Preprocess data if enabled
        if self.preprocess:
            X_processed = self._preprocess_data(X, fit=False)
        else:
            X_processed = X.copy()
        
        # Step 2: Generate features
        X_generated = self._generate_features(X_processed, fit=False)
        
        # Step 3: Select features
        X_selected = X_generated[self.selected_columns]
        
        return X_selected
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit the pipeline to the data and transform it.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        return self.fit(X, y).transform(X)
    
    def get_selected_features(self) -> List[str]:
        """Get the names of the selected features.
        
        Returns:
            List[str]: Names of the selected features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet. Call fit or fit_transform first.")
        return self.selected_columns
    
    def get_feature_importances(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dict: Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet. Call fit or fit_transform first.")
        return self.feature_importances_
    
    def get_runtime_stats(self) -> Dict[str, Any]:
        """Get runtime statistics.
        
        Returns:
            Dict: Dictionary with runtime statistics
        """
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet. Call fit or fit_transform first.")
        return self.runtime_stats_
    
    def get_evaluator(self) -> Optional[FeatureEvaluator]:
        """Get the feature evaluator.
        
        Returns:
            Optional[FeatureEvaluator]: The feature evaluator instance if available
        """
        return self.evaluator
    
    def save_pipeline(self, filepath: str) -> None:
        """Save the pipeline configuration to a JSON file.
        
        Args:
            filepath: Path to save the pipeline configuration
        """
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet. Call fit or fit_transform first.")
            
        # Create a configuration dictionary
        config = {
            'original_columns': self.original_columns,
            'generated_columns': self.generated_columns,
            'selected_columns': self.selected_columns,
            'feature_importances': {k: float(v) for k, v in self.feature_importances_.items()},
            'runtime_stats': self.runtime_stats_
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _preprocess_data(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Preprocess the data by handling missing values and scaling.
        
        Args:
            X: Input features
            fit: Whether to fit the preprocessors
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        X_processed = X.copy()
        
        if fit:
            # Initialize preprocessors
            self.numerical_imputer_ = SimpleImputer(strategy='mean')
            self.categorical_imputer_ = SimpleImputer(strategy='most_frequent')
            self.scaler_ = StandardScaler()
            
            # Get numerical and categorical columns
            self.numerical_columns_ = X.select_dtypes(include=['number']).columns.tolist()
            self.categorical_columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Impute missing values
        if hasattr(self, 'numerical_columns_') and self.numerical_columns_:
            numerical_cols = [col for col in self.numerical_columns_ if col in X.columns]
            if numerical_cols:
                if fit:
                    X_processed[numerical_cols] = self.numerical_imputer_.fit_transform(X_processed[numerical_cols])
                else:
                    X_processed[numerical_cols] = self.numerical_imputer_.transform(X_processed[numerical_cols])
        
        if hasattr(self, 'categorical_columns_') and self.categorical_columns_:
            categorical_cols = [col for col in self.categorical_columns_ if col in X.columns]
            if categorical_cols:
                # Convert to string first to handle different types of categorical data
                for col in categorical_cols:
                    X_processed[col] = X_processed[col].astype(str)
                
                if fit:
                    X_processed[categorical_cols] = self.categorical_imputer_.fit_transform(X_processed[categorical_cols])
                else:
                    X_processed[categorical_cols] = self.categorical_imputer_.transform(X_processed[categorical_cols])
        
        # Scale numerical features
        if hasattr(self, 'numerical_columns_') and self.numerical_columns_:
            numerical_cols = [col for col in self.numerical_columns_ if col in X.columns]
            if numerical_cols:
                if fit:
                    X_processed[numerical_cols] = self.scaler_.fit_transform(X_processed[numerical_cols])
                else:
                    X_processed[numerical_cols] = self.scaler_.transform(X_processed[numerical_cols])
        
        return X_processed
    
    def _generate_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Generate new features using the provided transformers.
        
        Args:
            X: Input features
            fit: Whether to fit the transformers
            
        Returns:
            pd.DataFrame: Data with generated features
        """
        X_result = X.copy()
        
        # Apply each feature generator
        for i, transformer in enumerate(self.generation_steps):
            if self.verbose > 0:
                print(f"Applying feature generator {i+1}/{len(self.generation_steps)}: {type(transformer).__name__}")
                
            try:
                if fit:
                    # Fit and transform
                    X_transformed = transformer.fit_transform(X_result)
                else:
                    # Transform only
                    X_transformed = transformer.transform(X_result)
                
                # Add new features to result
                new_cols = [col for col in X_transformed.columns if col not in X_result.columns]
                if new_cols:
                    X_result = pd.concat([X_result, X_transformed[new_cols]], axis=1)
                    
                    if self.verbose > 1:
                        print(f"  Added {len(new_cols)} new features: {new_cols[:5]}")
                        if len(new_cols) > 5:
                            print(f"  ... and {len(new_cols) - 5} more")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error in feature generator {type(transformer).__name__}: {str(e)}")
        
        return X_result
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Select the best features.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Tuple: Transformed data with selected features and list of selected column names
        """
        # Check if max_features is specified
        if self.max_features is not None:
            # Update the selector's max_features parameter if possible
            if hasattr(self.selection_method, 'max_features'):
                self.selection_method.max_features = self.max_features
            elif hasattr(self.selection_method, 'k'):
                self.selection_method.k = self.max_features
        
        # Check if target_metric is specified
        if self.target_metric is not None:
            # Update the selector's scoring parameter if possible
            if hasattr(self.selection_method, 'scoring'):
                self.selection_method.scoring = self.target_metric
                
        # Apply feature selection
        if self.verbose > 0:
            print(f"Selecting features with {type(self.selection_method).__name__}...")
            
        try:
            X_selected = self.selection_method.fit_transform(X, y)
            selected_columns = self.selection_method.get_selected_features()
            self.feature_importances_ = self.selection_method.get_feature_importances()
        except Exception as e:
            if self.verbose > 0:
                print(f"Error in feature selection: {str(e)}")
                print("Falling back to original features.")
            # Fallback
            selected_columns = list(X.columns)
            X_selected = X
            self.feature_importances_ = {col: 1.0 / len(X.columns) for col in X.columns}
        
        return X_selected, selected_columns
    
    def _evaluate_features(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Evaluate feature importance and impact on model performance.
        
        Args:
            X: Input features
            y: Target variable
        """
        if self.verbose > 0:
            print("Evaluating features...")
            
        try:
            # Initialize evaluator
            metrics = [self.target_metric] if self.target_metric else None
            self.evaluator = FeatureEvaluator(
                cv=5, 
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                metrics=metrics
            )
            
            # Fit the evaluator
            self.evaluator.fit(X, y)
            
            # Update feature importances with evaluation results
            evaluations = self.evaluator.evaluate_features()
            
            # Merge with existing importances
            for _, row in evaluations.iterrows():
                feature = row['feature']
                if feature in self.feature_importances_:
                    # Combine existing importance with evaluation
                    self.feature_importances_[feature] = (
                        self.feature_importances_[feature] + row['aggregate_importance']
                    ) / 2
        except Exception as e:
            if self.verbose > 0:
                print(f"Error in feature evaluation: {str(e)}")
                
    def __repr__(self) -> str:
        """Get a string representation of the pipeline.
        
        Returns:
            str: String representation
        """
        if not self.is_fitted:
            return f"FeaturePipeline(fitted=False, generation_steps={len(self.generation_steps)})"
        
        return (f"FeaturePipeline(fitted=True, "
                f"original_features={self.runtime_stats_.get('original_features', 'N/A')}, "
                f"generated_features={self.runtime_stats_.get('generated_features', 'N/A')}, "
                f"selected_features={self.runtime_stats_.get('selected_features', 'N/A')})")


# Collection of utility functions for the pipeline

def build_generation_pipeline(numerical: bool = True, 
                            categorical: bool = True, 
                            datetime: bool = True, 
                            text: bool = False,
                            interactions: bool = True,
                            feature_config: Dict[str, Dict[str, Any]] = None) -> List[BaseTransformer]:
    """Utility function to build a feature generation pipeline.
    
    Args:
        numerical: Whether to include numerical transformations
        categorical: Whether to include categorical transformations
        datetime: Whether to include datetime transformations
        text: Whether to include text transformations
        interactions: Whether to include feature interactions
        feature_config: Configuration for feature generators
        
    Returns:
        List[BaseTransformer]: List of feature generators
    """
    from autofeature.feature_generation import (
        MathematicalTransformer, InteractionTransformer, 
        AggregationTransformer, TimeBasedTransformer, TextBasedTransformer
    )
    
    # Default configurations
    default_config = {
        'mathematical': {
            'operations': ['square', 'log', 'sqrt'],
        },
        'interaction': {
            'interaction_types': ['multiplication'],
            'max_features': 100
        },
        'aggregation': {
            'aggregation_types': ['mean', 'median', 'min', 'max'],
        },
        'time_based': {
            'components': ['year', 'month', 'day', 'day_of_week', 'hour'],
            'cyclical': True
        },
        'text_based': {
            'features': ['char_count', 'word_count', 'special_char_count'],
            'include_tfidf': False
        }
    }
    
    # Merge with user config
    config = default_config.copy()
    if feature_config:
        for category, values in feature_config.items():
            if category in config:
                config[category].update(values)
            else:
                config[category] = values
    
    # Build generator list
    generators = []
    
    # Add numerical transformations
    if numerical:
        generators.append(MathematicalTransformer(**config.get('mathematical', {})))
    
    # Add interactions
    if interactions:
        generators.append(InteractionTransformer(**config.get('interaction', {})))
    
    # Add aggregations for categorical
    if categorical:
        generators.append(AggregationTransformer(**config.get('aggregation', {})))
    
    # Add datetime transformations
    if datetime:
        generators.append(TimeBasedTransformer(**config.get('time_based', {})))
    
    # Add text transformations
    if text:
        generators.append(TextBasedTransformer(**config.get('text_based', {})))
    
    return generators 