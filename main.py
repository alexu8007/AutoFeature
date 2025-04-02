#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the AutoFeature framework.
This script demonstrates how to use the framework with a Kaggle dataset.
"""

import os
import pandas as pd
import numpy as np
import zipfile
import requests
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.tree import Tree
from rich.text import Text
from rich.layout import Layout
from rich import box
from rich.columns import Columns
from autofeature.pipeline import FeaturePipeline
from autofeature.feature_generation import (
    MathematicalTransformer,
    InteractionTransformer,
    AggregationTransformer,
    TimeBasedTransformer
)
from autofeature.feature_selection import GeneticSelector

# Initialize Rich console
console = Console()


def download_housing_dataset():
    """Download California Housing dataset directly without using opendatasets."""
    with console.status("[bold green]Downloading California Housing dataset...", spinner="dots"):
        data_dir = "kaggle_data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Check if dataset already exists
        dataset_path = os.path.join(data_dir, "housing.csv")
        if os.path.exists(dataset_path):
            console.print("[green]Dataset already downloaded. Using local copy.[/green]")
            return dataset_path
        
        # Direct download link to the dataset (open source version)
        url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
        
        try:
            console.print(f"[bold yellow]Downloading dataset from[/bold yellow] [blue]{url}[/blue]")
            response = requests.get(url)
            response.raise_for_status()  # Check if download was successful
            
            # Save the dataset
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
            
            console.print(f"[bold green]Dataset successfully downloaded to[/bold green] [blue]{dataset_path}[/blue]")
            return dataset_path
            
        except Exception as e:
            console.print(f"[bold red]Error downloading dataset:[/bold red] {str(e)}")
            # Provide instructions for manual download as fallback
            console.print("\n[yellow]Please manually download the dataset:[/yellow]")
            console.print("1. Visit [link]https://www.kaggle.com/datasets/camnugent/california-housing-prices[/link]")
            console.print("2. Download and extract the housing.csv file")
            console.print(f"3. Place it in the [blue]{data_dir}[/blue] directory")
            return dataset_path


def main():
    """Run a demonstration of the AutoFeature framework with Kaggle data."""
    # Print header
    console.print(Panel.fit(
        "[bold blue]AutoFeature Framework[/bold blue] [bold yellow]with California Housing Dataset[/bold yellow]",
        border_style="green",
        title="Enterprise Demo",
        subtitle="Version 0.1.0"
    ))
    
    # Download and load the dataset
    data_file = download_housing_dataset()
    
    # Check if the file exists before trying to load it
    if not os.path.exists(data_file):
        console.print(f"[bold red]Error:[/bold red] Dataset file [blue]{data_file}[/blue] not found. Please download it manually.")
        return
    
    # Loading dataset with progress spinner
    with console.status("[bold green]Loading dataset...", spinner="dots"):
        housing_data = pd.read_csv(data_file).iloc[:1000]
    
    # Display dataset information
    console.print(f"\n[bold cyan]Dataset Information[/bold cyan]")
    info_table = Table(show_header=False, box=box.ROUNDED)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="yellow")
    info_table.add_row("Shape", f"{housing_data.shape[0]} rows × {housing_data.shape[1]} columns")
    info_table.add_row("Memory Usage", f"{housing_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    info_table.add_row("Missing Values", f"{housing_data.isna().sum().sum()} total")
    console.print(info_table)
    
    # Display dataset columns
    columns_table = Table(title="Dataset Columns", box=box.ROUNDED)
    columns_table.add_column("Column Name", style="cyan")
    columns_table.add_column("Data Type", style="green")
    columns_table.add_column("Non-Null Count", style="yellow")
    columns_table.add_column("Memory Usage", style="magenta")
    
    for col in housing_data.columns:
        columns_table.add_row(
            col,
            str(housing_data[col].dtype),
            f"{housing_data[col].count()} / {len(housing_data)}",
            f"{housing_data[col].memory_usage(deep=True) / 1024:.2f} KB"
        )
    
    console.print(columns_table)
    
    # Display sample data
    console.print("\n[bold cyan]Sample Data[/bold cyan]")
    sample_table = Table(box=box.ROUNDED)
    
    # Add columns
    for col in housing_data.columns:
        # Adjust column width based on content
        sample_table.add_column(col, overflow="fold")
    
    # Add rows
    for _, row in housing_data.head(5).iterrows():
        sample_table.add_row(*[str(val) for val in row.values])
    
    console.print(sample_table)
    
    # Add a date column for demonstration of time-based features
    with console.status("[bold green]Preparing data...", spinner="dots"):
        housing_data['date'] = pd.date_range(start='2020-01-01', periods=len(housing_data))
        
        # Prepare features and target
        X = housing_data.drop('median_house_value', axis=1)
        y = housing_data['median_house_value']
    
    # Create and run the pipeline with progress tracking
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
        generation_node.add("[green]MathematicalTransformer [dim](operations: square, log, sqrt)[/dim]")
        generation_node.add("[green]InteractionTransformer [dim](interactions: multiplication)[/dim]")
        generation_node.add("[green]AggregationTransformer [dim](groupby: ocean_proximity)[/dim]")
        generation_node.add("[green]TimeBasedTransformer [dim](datetime: date)[/dim]")
        
        selection_node = pipeline_tree.add("[bold cyan]Feature Selection")
        selection_node.add("[yellow]GeneticSelector [dim](n_generations: 5, population_size: 20)[/dim]")
        
        console.print(pipeline_tree)
        
        # Create pipeline
        task = progress.add_task("[green]Creating pipeline...", total=1)
        pipeline = FeaturePipeline(
            generation_steps=[
                MathematicalTransformer(operations=['square', 'log', 'sqrt']),
                InteractionTransformer(interaction_types=['multiplication']),
                AggregationTransformer(groupby_columns=['ocean_proximity']),
                TimeBasedTransformer(datetime_columns=['date'])
            ],
            selection_method=GeneticSelector(n_generations=5, population_size=20),
            target_metric='r2'
        )
        progress.update(task, advance=1)
        
        # Fit pipeline
        task = progress.add_task("[green]Fitting pipeline...", total=100)
        for i in range(0, 101, 5):
            time.sleep(0.1)  # Simulating work being done
            progress.update(task, completed=i)
        pipeline.fit(X, y)
        progress.update(task, completed=100)
        
        # Transform data
        task = progress.add_task("[green]Transforming data...", total=100)
        for i in range(0, 101, 10):
            time.sleep(0.05)  # Simulating work being done
            progress.update(task, completed=i)
        X_transformed = pipeline.transform(X)
        progress.update(task, completed=100)
    
    # Display results
    console.print("\n[bold cyan]Feature Engineering Results[/bold cyan]")
    
    results_table = Table(box=box.ROUNDED)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="yellow")
    results_table.add_row("Original Features", str(X.shape[1]))
    results_table.add_row("Generated Features", str(X_transformed.shape[1]))
    results_table.add_row("Selected Features", str(len(pipeline.get_selected_features())))
    console.print(results_table)
    
    # Display selected features
    console.print("\n[bold cyan]Selected Top Features[/bold cyan]")
    
    features = pipeline.get_selected_features()
    
    # Split features into 3 columns for display
    chunk_size = max(1, len(features) // 3)
    feature_chunks = [features[i:i + chunk_size] for i in range(0, len(features), chunk_size)]
    
    # Create panels for each chunk
    panels = []
    for chunk in feature_chunks:
        feature_list = "\n".join([f"[green]• [bold cyan]{feature}[/bold cyan][/green]" for feature in chunk])
        panels.append(Panel(feature_list, expand=True))
    
    # Create Columns with all panels
    feature_columns = Columns(panels)
    
    console.print(feature_columns)
    
    # Completion message
    console.print(Panel(
        "[bold green]Automated Feature Engineering completed successfully![/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main()
