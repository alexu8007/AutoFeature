"""
Feature Generation Module

This module contains transformers for generating new features through various
transformations of existing features.
"""

from autofeature.feature_generation.mathematical import MathematicalTransformer
from autofeature.feature_generation.interaction import InteractionTransformer
from autofeature.feature_generation.aggregation import AggregationTransformer
from autofeature.feature_generation.time_based import TimeBasedTransformer
from autofeature.feature_generation.text_based import TextBasedTransformer

__all__ = [
    'MathematicalTransformer',
    'InteractionTransformer',
    'AggregationTransformer',
    'TimeBasedTransformer',
    'TextBasedTransformer'
] 