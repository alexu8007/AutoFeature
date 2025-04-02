"""
Feature Selection Module

This module contains methods for selecting the most valuable features from a 
larger set of generated features.
"""

from autofeature.feature_selection.filter_methods import FilterSelector
from autofeature.feature_selection.wrapper_methods import WrapperSelector
from autofeature.feature_selection.embedded_methods import EmbeddedSelector
from autofeature.feature_selection.genetic_algorithm import GeneticSelector

__all__ = [
    'FilterSelector',
    'WrapperSelector',
    'EmbeddedSelector',
    'GeneticSelector'
] 