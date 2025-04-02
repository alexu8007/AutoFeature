"""
AutoFeature: Automated Feature Engineering Framework

This package provides tools and utilities for automated feature engineering
in machine learning pipelines, including feature generation, selection, and evaluation.
"""

__version__ = '0.1.0'
__author__ = 'AutoFeature Team'

from autofeature.feature_generation import (
    MathematicalTransformer,
    InteractionTransformer,
    AggregationTransformer,
    TimeBasedTransformer,
    TextBasedTransformer
)

from autofeature.feature_selection import (
    FilterSelector,
    WrapperSelector,
    EmbeddedSelector,
    GeneticSelector
)

from autofeature.feature_evaluation import (
    FeatureEvaluator
)

from autofeature.pipeline import (
    FeaturePipeline,
    build_generation_pipeline
) 