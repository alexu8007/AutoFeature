"""
Basic usage example for the AutoFeature framework.

This example demonstrates how to use the framework with a simple dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

from autofeature.pipeline import FeaturePipeline, build_generation_pipeline


def main():
    """Run the basic usage example."""
    print("AutoFeature Basic Usage Example")
    print("===============================")
    
    # Load California housing dataset
    print("\nLoading California housing dataset...")
    housing = fetch_california_housing()
    
    # Create a pandas DataFrame
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name="Price")
    
    # Add a datetime column for demonstration
    X['date'] = pd.date_range(start='2020-01-01', periods=len(X))
    
    # Examine data
    print(f"\nDataset shape: {X.shape}")
    print("\nFeatures:")
    print(X.head())
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create feature generators
    print("\nCreating feature generators...")
    generators = build_generation_pipeline(
        numerical=True,
        categorical=True,
        datetime=True,
        text=False,
        interactions=True,
        feature_config={
            'mathematical': {
                'operations': ['square', 'log', 'sqrt'],
            },
            'interaction': {
                'interaction_types': ['multiplication'],
                'max_features': 20
            },
        }
    )
    
    # Create pipeline
    print("\nCreating and running feature pipeline...")
    pipeline = FeaturePipeline(
        generation_steps=generators,
        target_metric='r2',
        max_features=15,
        verbose=1
    )
    
    # Fit and transform training data
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_transformed = pipeline.transform(X_test)
    
    # Check feature counts
    stats = pipeline.get_runtime_stats()
    print(f"\nOriginal features: {stats['original_features']}")
    print(f"Generated features: {stats['generated_features']}")
    print(f"Selected features: {stats['selected_features']}")
    
    # Print runtime stats
    print(f"\nRuntime statistics:")
    print(f"  - Feature generation time: {stats['generation_time']:.2f} seconds")
    print(f"  - Feature selection time: {stats['selection_time']:.2f} seconds")
    print(f"  - Total pipeline time: {stats['total_time']:.2f} seconds")
    
    # Print selected features
    print("\nSelected features:")
    for feature in pipeline.get_selected_features():
        print(f"  - {feature}")
    
    # Train a model with the original features
    print("\nTraining a model with original features...")
    rf_original = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_original.fit(X_train[housing.feature_names], y_train)
    
    # Evaluate on test set
    y_pred_original = rf_original.predict(X_test[housing.feature_names])
    r2_original = r2_score(y_test, y_pred_original)
    rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
    
    # Train a model with the transformed features
    print("\nTraining a model with transformed features...")
    rf_transformed = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_transformed.fit(X_train_transformed, y_train)
    
    # Evaluate on test set
    y_pred_transformed = rf_transformed.predict(X_test_transformed)
    r2_transformed = r2_score(y_test, y_pred_transformed)
    rmse_transformed = np.sqrt(mean_squared_error(y_test, y_pred_transformed))
    
    # Print results
    print("\nModel performance comparison:")
    print(f"  - Original features: R2 = {r2_original:.4f}, RMSE = {rmse_original:.4f}")
    print(f"  - Transformed features: R2 = {r2_transformed:.4f}, RMSE = {rmse_transformed:.4f}")
    
    # Get the evaluator and plot feature importances
    evaluator = pipeline.get_evaluator()
    if evaluator:
        print("\nPlotting feature importances...")
        plt.figure(figsize=(10, 6))
        evaluator.plot_feature_importances(top_n=10, method='aggregate')
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        print("Feature importances plot saved as 'feature_importances.png'")
        
        print("\nPlotting feature correlation matrix...")
        plt.figure(figsize=(12, 10))
        evaluator.plot_correlation_matrix(X_train_transformed)
        plt.tight_layout()
        plt.savefig('feature_correlation.png')
        print("Feature correlation plot saved as 'feature_correlation.png'")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 