"""
Pipeline Module

This module provides the main pipeline for automated feature engineering,
integrating feature generation, selection, and evaluation.
"""
from .pipeline import FeaturePipeline, build_generation_pipeline
from autofeature.pipeline.pipeline import FeaturePipeline

__all__ = ['FeaturePipeline'] 