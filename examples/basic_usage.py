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


def main():
    """Run the basic usage example."""
    # Print header
    console.print(Panel.fit(
        "[bold blue]AutoFeature Basic Usage Example[/bold blue]",
        border_style="green",
        title="Enterprise Demo",
        subtitle="Version 0.1.0"
    ))
    
    # Load California housing dataset
    with console.status("[bold green]Loading California housing dataset...", spinner="dots"):
        housing = fetch_california_housing()
        
        # Create a pandas DataFrame
        X = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = pd.Series(housing.target, name="Price")
        
        # Add a datetime column for demonstration
        X['date'] = pd.date_range(start='2020-01-01', periods=len(X))
    
    # Display dataset information
    console.print(f"\n[bold cyan]Dataset Information[/bold cyan]")
    info_table = Table(show_header=False, box=box.ROUNDED)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="yellow")
    info_table.add_row("Shape", f"{X.shape[0]} rows × {X.shape[1]} columns")
    info_table.add_row("Memory Usage", f"{X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    info_table.add_row("Missing Values", f"{X.isna().sum().sum()} total")
    console.print(info_table)
    
    # Display sample data
    console.print("\n[bold cyan]Sample Data[/bold cyan]")
    sample_table = Table(box=box.ROUNDED)
    
    # Add columns
    for col in X.columns[:5]:  # Limit to first 5 columns for display
        sample_table.add_column(col, overflow="fold")
    
    # Add rows
    for _, row in X.head(5).iterrows():
        sample_table.add_row(*[str(val) for val in row.values[:5]])
    
    console.print(sample_table)
    
    # Split the data
    with console.status("[bold green]Splitting data into train and test sets...", spinner="dots"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
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
        pipeline_tree = Tree("[bold blue]Pipeline Setup")
        
        generation_node = pipeline_tree.add("[bold cyan]Feature Generation")
        generation_node.add("[green]Mathematical Transformations [dim](square, log, sqrt)[/dim]")
        generation_node.add("[green]Feature Interactions [dim](multiplication)[/dim]")
        generation_node.add("[green]Datetime Features [dim](from date column)[/dim]")
        
        console.print(pipeline_tree)
        
        # Create feature generators
        task = progress.add_task("[green]Creating feature generators...", total=1)
        
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
        progress.update(task, advance=1)
        
        # Create pipeline
        task = progress.add_task("[green]Creating pipeline...", total=1)
        
        pipeline = FeaturePipeline(
            generation_steps=generators,
            target_metric='r2',
            max_features=15,
            verbose=1
        )
        progress.update(task, advance=1)
        
        # Fit and transform
        task = progress.add_task("[green]Fitting pipeline...", total=100)
        for i in range(0, 101, 5):
            time.sleep(0.05)  # Simulating work being done
            progress.update(task, completed=i)
        
        X_train_transformed = pipeline.fit_transform(X_train, y_train)
        progress.update(task, completed=100)
        
        # Transform test data
        task = progress.add_task("[green]Transforming test data...", total=100)
        for i in range(0, 101, 10):
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
    
    # Split features into 3 columns for display
    chunk_size = max(1, len(selected_features) // 3)
    feature_chunks = [selected_features[i:i + chunk_size] for i in range(0, len(selected_features), chunk_size)]
    
    # Create panels for each chunk
    panels = []
    for chunk in feature_chunks:
        feature_list = "\n".join([f"[green]• [bold cyan]{feature}[/bold cyan][/green]" for feature in chunk])
        panels.append(Panel(feature_list, expand=True))
    
    # Create Columns with all panels
    feature_columns = Columns(panels)
    console.print(feature_columns)
    
    # Model training and evaluation
    with console.status("[bold green]Training and evaluating models...", spinner="dots"):
        # Train a model with the original features
        rf_original = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_original.fit(X_train[housing.feature_names], y_train)
        
        # Evaluate on test set
        y_pred_original = rf_original.predict(X_test[housing.feature_names])
        r2_original = r2_score(y_test, y_pred_original)
        rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
        
        # Train a model with the transformed features
        rf_transformed = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_transformed.fit(X_train_transformed, y_train)
        
        # Evaluate on test set
        y_pred_transformed = rf_transformed.predict(X_test_transformed)
        r2_transformed = r2_score(y_test, y_pred_transformed)
        rmse_transformed = np.sqrt(mean_squared_error(y_test, y_pred_transformed))
    
    # Display model performance
    console.print("\n[bold cyan]Model Performance Comparison[/bold cyan]")
    performance_table = Table(box=box.ROUNDED)
    performance_table.add_column("Model", style="cyan")
    performance_table.add_column("R²", style="green")
    performance_table.add_column("RMSE", style="yellow")
    performance_table.add_column("Improvement", style="magenta")
    
    performance_table.add_row(
        "Original Features", 
        f"{r2_original:.4f}", 
        f"{rmse_original:.4f}",
        "-"
    )
    
    r2_improvement = ((r2_transformed - r2_original) / abs(r2_original)) * 100
    rmse_improvement = ((rmse_original - rmse_transformed) / rmse_original) * 100
    
    performance_table.add_row(
        "Transformed Features", 
        f"{r2_transformed:.4f}", 
        f"{rmse_transformed:.4f}",
        f"{rmse_improvement:.1f}% RMSE reduction"
    )
    
    console.print(performance_table)
    
    # Get the evaluator and plot feature importances
    evaluator = pipeline.get_evaluator()
    if evaluator:
        with console.status("[bold green]Plotting feature importances...", spinner="dots"):
            plt.figure(figsize=(10, 6))
            evaluator.plot_feature_importances(top_n=10, method='aggregate')
            plt.tight_layout()
            plt.savefig('feature_importances.png')
        console.print("[green]Feature importances plot saved as 'feature_importances.png'[/green]")
        
        with console.status("[bold green]Plotting feature correlation matrix...", spinner="dots"):
            plt.figure(figsize=(12, 10))
            evaluator.plot_correlation_matrix(X_train_transformed)
            plt.tight_layout()
            plt.savefig('feature_correlation.png')
        console.print("[green]Feature correlation plot saved as 'feature_correlation.png'[/green]")
    
    # Completion message
    console.print(Panel(
        "[bold green]Example completed successfully![/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main() 