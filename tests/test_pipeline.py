"""
Tests for the feature pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from autofeature.pipeline import FeaturePipeline, build_generation_pipeline
from autofeature.feature_generation import (
    MathematicalTransformer,
    InteractionTransformer,
    TimeBasedTransformer
)
from autofeature.feature_selection import (
    FilterSelector,
    GeneticSelector
)


@pytest.fixture
def regression_data():
    """Create a regression dataset for testing."""
    X, y = make_regression(n_samples=100, n_features=5, n_informative=3, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(X.shape[1])])
    y = pd.Series(y, name='target')
    
    # Add a datetime column
    X['date'] = pd.date_range(start='2020-01-01', periods=100)
    
    return X, y


@pytest.fixture
def classification_data():
    """Create a classification dataset for testing."""
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, 
                              n_redundant=1, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(X.shape[1])])
    y = pd.Series(y, name='target')
    
    # Add a categorical column
    X['category'] = np.random.choice(['A', 'B', 'C'], size=100)
    
    return X, y


def test_pipeline_initialization():
    """Test pipeline initialization with different parameters."""
    # Default initialization
    pipeline = FeaturePipeline()
    assert pipeline.generation_steps == []
    assert pipeline.is_fitted == False
    
    # Custom initialization
    generators = [MathematicalTransformer(), InteractionTransformer()]
    selector = FilterSelector(method='f_regression', k=3)
    
    pipeline = FeaturePipeline(
        generation_steps=generators,
        selection_method=selector,
        target_metric='r2',
        max_features=5,
        verbose=1
    )
    
    assert len(pipeline.generation_steps) == 2
    assert pipeline.target_metric == 'r2'
    assert pipeline.max_features == 5


def test_pipeline_regression(regression_data):
    """Test pipeline with regression data."""
    X, y = regression_data
    
    # Build generators
    generators = build_generation_pipeline(
        numerical=True,
        categorical=False,
        datetime=True,
        text=False,
        interactions=True
    )
    
    # Create pipeline
    pipeline = FeaturePipeline(
        generation_steps=generators,
        selection_method=GeneticSelector(n_generations=2, population_size=10),
        max_features=3,
        verbose=0
    )
    
    # Fit and transform
    X_transformed = pipeline.fit_transform(X, y)
    
    # Check results
    assert pipeline.is_fitted
    assert len(pipeline.original_columns) == X.shape[1]
    assert len(pipeline.generated_columns) > 0
    assert len(pipeline.selected_columns) <= 3
    assert X_transformed.shape[1] <= 3
    
    # Test transform only
    X_test_transformed = pipeline.transform(X)
    assert X_test_transformed.shape[1] == X_transformed.shape[1]
    
    # Test feature importances
    importances = pipeline.get_feature_importances()
    assert len(importances) > 0
    assert max(importances.values()) <= 1.0
    
    # Test runtime stats
    stats = pipeline.get_runtime_stats()
    assert 'original_features' in stats
    assert 'generated_features' in stats
    assert 'selected_features' in stats
    assert 'total_time' in stats


def test_pipeline_classification(classification_data):
    """Test pipeline with classification data."""
    X, y = classification_data
    
    # Build generators
    generators = [
        MathematicalTransformer(operations=['square', 'sqrt']),
        InteractionTransformer(interaction_types=['multiplication'])
    ]
    
    # Create pipeline
    pipeline = FeaturePipeline(
        generation_steps=generators,
        selection_method=FilterSelector(method='f_classif', k=3),
        verbose=0
    )
    
    # Fit and transform
    X_transformed = pipeline.fit_transform(X, y)
    
    # Check results
    assert pipeline.is_fitted
    assert len(pipeline.selected_columns) <= 3
    
    # Get selected features
    selected = pipeline.get_selected_features()
    assert isinstance(selected, list)
    assert len(selected) <= 3


def test_pipeline_errors():
    """Test pipeline error handling."""
    # Create pipeline
    pipeline = FeaturePipeline()
    
    # Test with non-DataFrame input
    X = np.random.random((10, 3))
    y = np.random.random(10)
    
    with pytest.raises(ValueError):
        pipeline.fit(X, y)
    
    # Test transform before fit
    X_df = pd.DataFrame(X, columns=['a', 'b', 'c'])
    
    with pytest.raises(ValueError):
        pipeline.transform(X_df)
    
    with pytest.raises(ValueError):
        pipeline.get_selected_features()


def test_build_generation_pipeline():
    """Test the build_generation_pipeline utility function."""
    # Test with default parameters
    generators = build_generation_pipeline()
    assert len(generators) == 4  # numerical, interactions, categorical, datetime
    
    # Test with custom parameters
    generators = build_generation_pipeline(
        numerical=True,
        categorical=False,
        datetime=False,
        text=True,
        interactions=False
    )
    assert len(generators) == 2  # numerical, text
    
    # Test with custom config
    feature_config = {
        'mathematical': {
            'operations': ['square', 'cube']
        },
        'text_based': {
            'include_tfidf': True
        }
    }
    
    generators = build_generation_pipeline(
        numerical=True,
        text=True,
        feature_config=feature_config
    )
    
    # Check the mathematical transformer
    math_transformer = generators[0]
    assert isinstance(math_transformer, MathematicalTransformer)
    assert math_transformer.operations == ['square', 'cube'] 