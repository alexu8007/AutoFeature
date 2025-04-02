"""
Time Series Feature Engineering Example for the AutoFeature framework.

This example demonstrates how to use the framework with time series data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from autofeature.pipeline import FeaturePipeline, build_generation_pipeline


def generate_time_series_data(n_samples=1000, freq='D'):
    """Generate synthetic time series data for this example."""
    np.random.seed(42)
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq=freq)
    
    # Base signal components
    time_index = np.arange(n_samples)
    
    # Trend component
    trend = 0.01 * time_index
    
    # Seasonal component (multiple seasonal patterns)
    season1 = 5 * np.sin(2 * np.pi * time_index / 365.25)  # yearly seasonality
    season2 = 2 * np.sin(2 * np.pi * time_index / 30.5)    # monthly seasonality
    season3 = 1 * np.sin(2 * np.pi * time_index / 7)       # weekly seasonality
    
    # Special events (e.g., holidays)
    special_events = np.zeros(n_samples)
    # Add some holiday effects (simplified)
    holiday_indices = [i for i in range(n_samples) if (i % 365) in [0, 180, 359]]  # New Year, mid-year, Christmas
    special_events[holiday_indices] = 10
    
    # Create some additional features that may be relevant
    temperature = 20 + 15 * np.sin(2 * np.pi * time_index / 365.25) + np.random.normal(0, 2, n_samples)
    precipitation = np.random.exponential(5, n_samples) * (1 + 0.5 * np.sin(2 * np.pi * time_index / 365.25))
    
    # Combine components with some noise
    target = trend + season1 + season2 + season3 + special_events + np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'precipitation': precipitation,
        'target': target
    })
    
    # Add some missing values to make it more realistic
    mask = np.random.random(n_samples) < 0.01  # 1% missing values
    df.loc[mask, 'temperature'] = np.nan
    
    return df


def main():
    """Run the time series feature engineering example."""
    print("AutoFeature Time Series Feature Engineering Example")
    print("=================================================")
    
    # Generate synthetic time series data
    print("\nGenerating synthetic time series data...")
    df = generate_time_series_data(n_samples=1095)  # about 3 years of daily data
    
    # Examine data
    print(f"\nDataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    
    # Plot the time series
    print("\nPlotting the time series data...")
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(df['date'], df['target'], label='Target')
    plt.legend()
    plt.title('Target Variable')
    
    plt.subplot(3, 1, 2)
    plt.plot(df['date'], df['temperature'], label='Temperature')
    plt.legend()
    plt.title('Temperature')
    
    plt.subplot(3, 1, 3)
    plt.plot(df['date'], df['precipitation'], label='Precipitation')
    plt.legend()
    plt.title('Precipitation')
    
    plt.tight_layout()
    plt.savefig('time_series_data.png')
    print("Time series plot saved as 'time_series_data.png'")
    
    # Prepare data for modeling
    # For time series, we use a time-based split instead of random
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    
    # Separate features and target
    X_train = df_train.drop('target', axis=1)
    y_train = df_train['target']
    X_test = df_test.drop('target', axis=1)
    y_test = df_test['target']
    
    # Create feature generators with time-based features enabled
    print("\nCreating feature generators with time-based processing...")
    generators = build_generation_pipeline(
        numerical=True,
        categorical=False,
        datetime=True,
        text=False,
        interactions=True,
        feature_config={
            'mathematical': {
                'operations': ['square', 'log', 'sqrt', 'abs'],
            },
            'time': {
                'datetime_columns': ['date'],
                'extract_components': True,
                'cyclical_encoding': True,
                'create_lags': True,
                'lag_values': [1, 7, 14, 30],
                'rolling_windows': [7, 14, 30],
                'rolling_functions': ['mean', 'std', 'min', 'max']
            },
            'interaction': {
                'interaction_types': ['multiplication', 'division'],
                'max_features': 20
            },
        }
    )
    
    # Create pipeline with time series cross-validation
    print("\nCreating and running feature pipeline...")
    time_cv = TimeSeriesSplit(n_splits=3)
    
    pipeline = FeaturePipeline(
        generation_steps=generators,
        selection_method='wrapper',
        selection_params={
            'model_type': 'regression',
            'model_name': 'random_forest',
            'scoring': 'neg_mean_squared_error',
            'cv': time_cv,
            'direction': 'forward'
        },
        target_metric='neg_mean_squared_error',
        max_features=20,
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
    
    # Train baseline model (using only original features)
    print("\nTraining baseline model (original features only)...")
    rf_baseline = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Handle missing values for baseline model
    X_train_baseline = X_train.copy()
    X_test_baseline = X_test.copy()
    for col in X_train_baseline.columns:
        if X_train_baseline[col].dtype in [np.float64, np.float32]:
            X_train_baseline[col] = X_train_baseline[col].fillna(X_train_baseline[col].mean())
            X_test_baseline[col] = X_test_baseline[col].fillna(X_train_baseline[col].mean())
    
    rf_baseline.fit(X_train_baseline, y_train)
    y_pred_baseline = rf_baseline.predict(X_test_baseline)
    
    # Train model with transformed features
    print("\nTraining model with transformed features...")
    rf_transformed = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_transformed.fit(X_train_transformed, y_train)
    y_pred_transformed = rf_transformed.predict(X_test_transformed)
    
    # Calculate metrics
    rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    r2_baseline = r2_score(y_test, y_pred_baseline)
    
    rmse_transformed = np.sqrt(mean_squared_error(y_test, y_pred_transformed))
    r2_transformed = r2_score(y_test, y_pred_transformed)
    
    # Print results
    print("\nModel performance comparison:")
    print(f"  - Baseline (original features): RMSE = {rmse_baseline:.4f}, R² = {r2_baseline:.4f}")
    print(f"  - Enhanced (transformed features): RMSE = {rmse_transformed:.4f}, R² = {r2_transformed:.4f}")
    print(f"  - Improvement: {((rmse_baseline - rmse_transformed) / rmse_baseline * 100):.2f}% reduction in RMSE")
    
    # Plot predictions
    print("\nPlotting predictions...")
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['date'], y_test, label='Actual', alpha=0.7)
    plt.plot(df_test['date'], y_pred_baseline, label='Baseline Prediction', alpha=0.7)
    plt.plot(df_test['date'], y_pred_transformed, label='Enhanced Prediction', alpha=0.7)
    plt.legend()
    plt.title('Time Series Predictions')
    plt.tight_layout()
    plt.savefig('time_series_predictions.png')
    print("Predictions plot saved as 'time_series_predictions.png'")
    
    # Get the evaluator and plot feature importances
    evaluator = pipeline.get_evaluator()
    if evaluator:
        print("\nPlotting feature importances...")
        plt.figure(figsize=(12, 8))
        evaluator.plot_feature_importances(top_n=15, method='aggregate')
        plt.tight_layout()
        plt.savefig('time_series_feature_importances.png')
        print("Feature importances plot saved as 'time_series_feature_importances.png'")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 