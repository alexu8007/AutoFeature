"""
Command-line interface for the AutoFeature framework.

This module provides a command-line interface to use the AutoFeature framework
for automated feature engineering tasks.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from autofeature.pipeline import FeaturePipeline, build_generation_pipeline
from autofeature.utils.io import load_data, save_data, serialize_cv


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AutoFeature: Automated Feature Engineering Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input data file (CSV, Excel, or Parquet)')
    parser.add_argument('--output', type=str, default='autofeature_output',
                        help='Directory to save output files')
    parser.add_argument('--target', type=str, required=True,
                        help='Name of the target column')
    parser.add_argument('--task', type=str, choices=['auto', 'regression', 'classification'],
                        default='auto', help='Type of machine learning task')
    
    # Feature generation
    gen_group = parser.add_argument_group('Feature Generation')
    gen_group.add_argument('--numerical', action='store_true', default=True,
                          help='Enable numerical feature transformations')
    gen_group.add_argument('--categorical', action='store_true', default=True,
                          help='Enable categorical feature transformations')
    gen_group.add_argument('--datetime', action='store_true', default=True,
                          help='Enable datetime feature transformations')
    gen_group.add_argument('--text', action='store_true', default=False,
                          help='Enable text feature transformations')
    gen_group.add_argument('--interactions', action='store_true', default=True,
                          help='Enable feature interactions')
    gen_group.add_argument('--datetime-cols', type=str, nargs='+',
                          help='Columns to treat as datetime. If not specified, auto-detection is used')
    gen_group.add_argument('--text-cols', type=str, nargs='+',
                          help='Columns to treat as text. If not specified, auto-detection is used')
    gen_group.add_argument('--categorical-cols', type=str, nargs='+',
                          help='Columns to treat as categorical. If not specified, auto-detection is used')
    
    # Feature selection
    sel_group = parser.add_argument_group('Feature Selection')
    sel_group.add_argument('--selection-method', type=str, 
                          choices=['filter', 'wrapper', 'embedded', 'genetic'],
                          default='embedded', help='Feature selection method')
    sel_group.add_argument('--max-features', type=int, default=None,
                          help='Maximum number of features to select')
    sel_group.add_argument('--model', type=str, 
                          choices=['random_forest', 'gradient_boosting', 'logistic_regression', 'linear_regression'],
                          default='random_forest', help='Model to use for feature selection')
    sel_group.add_argument('--cv', type=int, default=5,
                          help='Number of cross-validation folds')
    sel_group.add_argument('--scoring', type=str, 
                          help='Scoring metric for model evaluation (defaults to r2 for regression, accuracy for classification)')
    
    # Other settings
    other_group = parser.add_argument_group('Other Settings')
    other_group.add_argument('--config', type=str,
                           help='Path to JSON configuration file')
    other_group.add_argument('--save-pipeline', action='store_true', default=True,
                           help='Save the fitted pipeline')
    other_group.add_argument('--save-transformed', action='store_true', default=True,
                           help='Save the transformed dataset')
    other_group.add_argument('--verbose', type=int, default=1,
                           help='Verbosity level (0: quiet, 1: normal, 2: detailed)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def prepare_generation_config(args, config=None):
    """Prepare feature generation configuration."""
    generation_config = {
        'numerical': args.numerical,
        'categorical': args.categorical,
        'datetime': args.datetime,
        'text': args.text,
        'interactions': args.interactions,
        'feature_config': {}
    }
    
    # Override with config file if provided
    if config and 'generation' in config:
        for key, value in config['generation'].items():
            if key == 'feature_config':
                generation_config['feature_config'].update(value)
            else:
                generation_config[key] = value
    
    # Add specified columns
    if args.datetime_cols:
        if 'time' not in generation_config['feature_config']:
            generation_config['feature_config']['time'] = {}
        generation_config['feature_config']['time']['datetime_columns'] = args.datetime_cols
    
    if args.text_cols:
        if 'text' not in generation_config['feature_config']:
            generation_config['feature_config']['text'] = {}
        generation_config['feature_config']['text']['text_columns'] = args.text_cols
    
    if args.categorical_cols:
        if 'categorical' not in generation_config['feature_config']:
            generation_config['feature_config']['categorical'] = {}
        generation_config['feature_config']['categorical']['categorical_columns'] = args.categorical_cols
    
    return generation_config


def prepare_selection_config(args, config=None):
    """Prepare feature selection configuration."""
    selection_config = {
        'selection_method': args.selection_method,
        'selection_params': {
            'model_type': args.task,
            'model_name': args.model,
            'cv': args.cv
        },
        'max_features': args.max_features
    }
    
    # Add scoring if specified
    if args.scoring:
        selection_config['selection_params']['scoring'] = args.scoring
    
    # Override with config file if provided
    if config and 'selection' in config:
        for key, value in config['selection'].items():
            if key == 'selection_params':
                selection_config['selection_params'].update(value)
            else:
                selection_config[key] = value
    
    return selection_config


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Load config file if provided
    config = None
    if args.config:
        try:
            config = load_config(args.config)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load data
    try:
        data = load_data(args.input)
        print(f"Loaded data from {args.input}: {data.shape[0]} rows, {data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)
    
    # Check if target exists in data
    if args.target not in data.columns:
        print(f"Error: Target column '{args.target}' not found in data")
        print(f"Available columns: {', '.join(data.columns)}")
        sys.exit(1)
    
    # Split features and target
    X = data.drop(columns=[args.target])
    y = data[args.target]
    
    # Get generation configuration
    generation_config = prepare_generation_config(args, config)
    
    # Build generation pipeline
    print(f"Building feature generation pipeline...")
    generators = build_generation_pipeline(**generation_config)
    
    # Get selection configuration
    selection_config = prepare_selection_config(args, config)
    
    # Create the pipeline
    print(f"Creating feature pipeline with {selection_config['selection_method']} selection method...")
    pipeline = FeaturePipeline(
        generation_steps=generators,
        **selection_config,
        target_metric=args.scoring,
        verbose=args.verbose
    )
    
    # Fit and transform
    print(f"Fitting pipeline to data...")
    start_time = datetime.now()
    X_transformed = pipeline.fit_transform(X, y)
    end_time = datetime.now()
    
    # Get statistics
    stats = pipeline.get_runtime_stats()
    
    # Print results
    print("\nAutoFeature Results:")
    print(f"Original features: {stats['original_features']}")
    print(f"Generated features: {stats['generated_features']}")
    print(f"Selected features: {stats['selected_features']}")
    print(f"Total runtime: {(end_time - start_time).total_seconds():.2f} seconds")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save transformed data
    if args.save_transformed:
        output_file = os.path.join(args.output, f"transformed_data_{timestamp}.csv")
        transformed_with_target = pd.concat([X_transformed, y], axis=1)
        save_data(transformed_with_target, output_file)
        print(f"Transformed data saved to {output_file}")
    
    # Save pipeline
    if args.save_pipeline:
        pipeline_file = os.path.join(args.output, f"pipeline_{timestamp}.pkl")
        pipeline.save_pipeline(pipeline_file)
        print(f"Pipeline saved to {pipeline_file}")
    
    # Save feature list
    features_file = os.path.join(args.output, f"selected_features_{timestamp}.json")
    with open(features_file, 'w') as f:
        json.dump({
            'original_features': list(X.columns),
            'selected_features': pipeline.get_selected_features(),
            'feature_importances': pipeline.get_feature_importances()
        }, f, indent=2)
    print(f"Feature information saved to {features_file}")
    
    # Save runtime stats
    stats_file = os.path.join(args.output, f"runtime_stats_{timestamp}.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Runtime statistics saved to {stats_file}")
    
    print("\nAutoFeature process completed successfully.")


if __name__ == "__main__":
    main() 