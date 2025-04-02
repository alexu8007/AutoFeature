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
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.tree import Tree
from rich import box
from rich.columns import Columns

from autofeature.pipeline import FeaturePipeline, build_generation_pipeline

# Initialize Rich console
console = Console()


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
    # Print header
    console.print(Panel.fit(
        "[bold blue]AutoFeature Time Series Feature Engineering Example[/bold blue]",
        border_style="green",
        title="Enterprise Demo",
        subtitle="Version 0.1.0"
    ))
    
    # Generate synthetic time series data
    with console.status("[bold green]Generating synthetic time series data...", spinner="dots"):
        df = generate_time_series_data(n_samples=1095)  # about 3 years of daily data
    
    # Display dataset information
    console.print(f"\n[bold cyan]Dataset Information[/bold cyan]")
    info_table = Table(show_header=False, box=box.ROUNDED)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="yellow")
    info_table.add_row("Shape", f"{df.shape[0]} rows × {df.shape[1]} columns")
    info_table.add_row("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
    info_table.add_row("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    info_table.add_row("Missing Values", f"{df.isna().sum().sum()} cells")
    console.print(info_table)
    
    # Display sample data
    console.print("\n[bold cyan]Sample Data[/bold cyan]")
    sample_table = Table(box=box.ROUNDED)
    
    for col in df.columns:
        sample_table.add_column(col, overflow="fold")
    
    for _, row in df.head(5).iterrows():
        sample_table.add_row(
            str(row['date']),
            f"{row['temperature']:.2f}",
            f"{row['precipitation']:.2f}",
            f"{row['target']:.2f}"
        )
    
    console.print(sample_table)
    
    # Plot the time series
    with console.status("[bold green]Plotting the time series data...", spinner="dots"):
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
    console.print("[green]Time series plot saved as 'time_series_data.png'[/green]")
    
    # Prepare data for modeling
    with console.status("[bold green]Preparing data for modeling...", spinner="dots"):
        # For time series, we use a time-based split instead of random
        train_size = int(len(df) * 0.8)
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]
        
        # Separate features and target
        X_train = df_train.drop('target', axis=1)
        y_train = df_train['target']
        X_test = df_test.drop('target', axis=1)
        y_test = df_test['target']
    
    # Create pipeline with progress tracking
    console.print("\n[bold cyan]Feature Engineering Pipeline[/bold cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("[bold yellow]{task.percentage:.0f}%"),
        TimeElapsedColumn()
    ) as progress:
        # Display the pipeline components
        pipeline_tree = Tree("[bold blue]Pipeline Components")
        
        generation_node = pipeline_tree.add("[bold cyan]Feature Generation")
        generation_node.add("[green]Mathematical Transformations [dim](square, log, sqrt, abs)[/dim]")
        generation_node.add("[green]Feature Interactions [dim](multiplication, division)[/dim]")
        time_node = generation_node.add("[green]Time-based Features")
        time_node.add("[blue]Components: [dim]year, month, day, dayofweek, etc.[/dim]")
        time_node.add("[blue]Lags: [dim]1, 7, 14, 30 days[/dim]")
        time_node.add("[blue]Rolling windows: [dim]7, 14, 30 days (mean, std, min, max)[/dim]")
        
        selection_node = pipeline_tree.add("[bold cyan]Feature Selection")
        selection_node.add("[yellow]Wrapper Selector [dim](Random Forest, TimeSeriesSplit CV)[/dim]")
        
        console.print(pipeline_tree)
        
        # Create feature generators with time-based features enabled
        task = progress.add_task("[green]Creating feature generators...", total=1)
        
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
        progress.update(task, advance=1)
        
        # Create pipeline with time series cross-validation
        task = progress.add_task("[green]Setting up time series pipeline...", total=1)
        
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
        progress.update(task, advance=1)
        
        # Fit and transform
        task = progress.add_task("[green]Fitting pipeline (this may take a while)...", total=100)
        for i in range(0, 101, 2):
            time.sleep(0.05)  # Simulating work being done
            progress.update(task, completed=i)
        
        X_train_transformed = pipeline.fit_transform(X_train, y_train)
        progress.update(task, completed=100)
        
        # Transform test data
        task = progress.add_task("[green]Transforming test data...", total=100)
        for i in range(0, 101, 5):
            time.sleep(0.02)  # Simulating work being done
            progress.update(task, completed=i)
        
        X_test_transformed = pipeline.transform(X_test)
        progress.update(task, completed=100)
    
    # Display feature engineering statistics
    stats = pipeline.get_runtime_stats()
    
    console.print("\n[bold cyan]Feature Engineering Statistics[/bold cyan]")
    stats_table = Table(box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="yellow")
    stats_table.add_row("Original Features", str(stats['original_features']))
    stats_table.add_row("Generated Features", str(stats['generated_features']))
    stats_table.add_row("Selected Features", str(stats['selected_features']))
    stats_table.add_row("Feature Generation Time", f"{stats['generation_time']:.2f} seconds")
    stats_table.add_row("Feature Selection Time", f"{stats['selection_time']:.2f} seconds")
    stats_table.add_row("Total Pipeline Time", f"{stats['total_time']:.2f} seconds")
    console.print(stats_table)
    
    # Display selected features
    console.print("\n[bold cyan]Selected Features[/bold cyan]")
    selected_features = pipeline.get_selected_features()
    
    # Group features by type for better presentation
    time_features = [f for f in selected_features if any(t in f for t in ['date_', 'lag_', 'rolling_'])]
    math_features = [f for f in selected_features if any(t in f for t in ['_square', '_log', '_sqrt', '_abs'])]
    interaction_features = [f for f in selected_features if any(t in f for t in ['_multiplication_', '_division_'])]
    original_features = [f for f in selected_features if f not in time_features + math_features + interaction_features]
    
    # Create panels for each feature type
    panels = []
    
    if original_features:
        feature_list = "\n".join([f"[green]• [bold cyan]{feature}[/bold cyan][/green]" for feature in original_features])
        panels.append(Panel(feature_list, title="[yellow]Original Features[/yellow]", expand=True))
    
    if time_features:
        feature_list = "\n".join([f"[green]• [bold cyan]{feature}[/bold cyan][/green]" for feature in time_features[:10]])
        if len(time_features) > 10:
            feature_list += f"\n[dim]... and {len(time_features) - 10} more[/dim]"
        panels.append(Panel(feature_list, title="[yellow]Time Features[/yellow]", expand=True))
    
    if math_features:
        feature_list = "\n".join([f"[green]• [bold cyan]{feature}[/bold cyan][/green]" for feature in math_features])
        panels.append(Panel(feature_list, title="[yellow]Mathematical Features[/yellow]", expand=True))
    
    if interaction_features:
        feature_list = "\n".join([f"[green]• [bold cyan]{feature}[/bold cyan][/green]" for feature in interaction_features])
        panels.append(Panel(feature_list, title="[yellow]Interaction Features[/yellow]", expand=True))
    
    # Create Columns with all panels
    feature_columns = Columns(panels)
    console.print(feature_columns)
    
    # Model training and evaluation
    with console.status("[bold green]Training and evaluating models...", spinner="dots"):
        # Train baseline model (using only original features)
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
        rf_transformed = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_transformed.fit(X_train_transformed, y_train)
        y_pred_transformed = rf_transformed.predict(X_test_transformed)
        
        # Calculate metrics
        rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
        r2_baseline = r2_score(y_test, y_pred_baseline)
        
        rmse_transformed = np.sqrt(mean_squared_error(y_test, y_pred_transformed))
        r2_transformed = r2_score(y_test, y_pred_transformed)
    
    # Display model performance
    console.print("\n[bold cyan]Model Performance Comparison[/bold cyan]")
    performance_table = Table(box=box.ROUNDED)
    performance_table.add_column("Model", style="cyan")
    performance_table.add_column("RMSE", style="green")
    performance_table.add_column("R²", style="yellow")
    performance_table.add_column("Improvement", style="magenta")
    
    performance_table.add_row(
        "Baseline (Original Features)", 
        f"{rmse_baseline:.4f}", 
        f"{r2_baseline:.4f}",
        "-"
    )
    
    rmse_improvement = ((rmse_baseline - rmse_transformed) / rmse_baseline * 100)
    r2_improvement = ((r2_transformed - r2_baseline) / max(0.001, abs(r2_baseline))) * 100
    
    performance_table.add_row(
        "Enhanced (Transformed Features)", 
        f"{rmse_transformed:.4f}", 
        f"{r2_transformed:.4f}",
        f"{rmse_improvement:.1f}% RMSE reduction"
    )
    
    console.print(performance_table)
    
    # Plot predictions
    with console.status("[bold green]Plotting predictions...", spinner="dots"):
        plt.figure(figsize=(12, 6))
        plt.plot(df_test['date'], y_test, label='Actual', alpha=0.7)
        plt.plot(df_test['date'], y_pred_baseline, label='Baseline Prediction', alpha=0.7)
        plt.plot(df_test['date'], y_pred_transformed, label='Enhanced Prediction', alpha=0.7)
        plt.legend()
        plt.title('Time Series Predictions')
        plt.tight_layout()
        plt.savefig('time_series_predictions.png')
    console.print("[green]Predictions plot saved as 'time_series_predictions.png'[/green]")
    
    # Get the evaluator and plot feature importances
    evaluator = pipeline.get_evaluator()
    if evaluator:
        with console.status("[bold green]Plotting feature importances...", spinner="dots"):
            plt.figure(figsize=(12, 8))
            evaluator.plot_feature_importances(top_n=15, method='aggregate')
            plt.tight_layout()
            plt.savefig('time_series_feature_importances.png')
        console.print("[green]Feature importances plot saved as 'time_series_feature_importances.png'[/green]")
    
    # Completion message
    console.print(Panel(
        "[bold green]Time Series Feature Engineering Example completed successfully![/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main() 