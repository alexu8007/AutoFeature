"""
Text Feature Engineering Example for the AutoFeature framework.

This example demonstrates how to use the framework with text data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    """Run the text feature engineering example."""
    # Print header
    console.print(Panel.fit(
        "[bold blue]AutoFeature Text Feature Engineering Example[/bold blue]",
        border_style="green",
        title="Enterprise Demo",
        subtitle="Version 0.1.0"
    ))
    
    # Load dataset
    with console.status("[bold green]Loading 20 newsgroups dataset (subset)...", spinner="dots"):
        # Load a subset of the 20 newsgroups dataset (3 categories)
        categories = ['alt.atheism', 'comp.graphics', 'sci.med']
        newsgroups = fetch_20newsgroups(
            subset='all',
            categories=categories,
            remove=('headers', 'footers', 'quotes')
        )
        
        # Create a pandas DataFrame
        df = pd.DataFrame({
            'text': newsgroups.data,
            'category': newsgroups.target
        })
        
        # Add a numeric column for demonstration of mixed data types
        df['importance'] = np.random.randint(1, 10, size=len(df))
        
        # Clean up text and limit dataset size for faster processing
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df = df.sample(1000, random_state=42).reset_index(drop=True)
        
        # Map targets to category names
        category_names = {i: name for i, name in enumerate(categories)}
        df['category_name'] = df['category'].map(category_names)
    
    # Display dataset information
    console.print(f"\n[bold cyan]Dataset Information[/bold cyan]")
    info_table = Table(show_header=False, box=box.ROUNDED)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="yellow")
    info_table.add_row("Shape", f"{df.shape[0]} rows × {df.shape[1]} columns")
    info_table.add_row("Categories", f"{len(categories)} ({', '.join(categories)})")
    info_table.add_row("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    console.print(info_table)
    
    # Display sample data
    console.print("\n[bold cyan]Sample Data[/bold cyan]")
    
    # For text data, we'll create a special table with truncated text
    sample_table = Table(box=box.ROUNDED)
    sample_table.add_column("Category", style="green")
    sample_table.add_column("Importance", style="yellow")
    sample_table.add_column("Text Sample", style="cyan", overflow="fold", max_width=60)
    
    for _, row in df[['category_name', 'importance', 'text']].head(3).iterrows():
        sample_table.add_row(
            row['category_name'],
            str(row['importance']),
            row['text'][:200] + "..." if len(row['text']) > 200 else row['text']
        )
    
    console.print(sample_table)
    
    # Split features and target
    with console.status("[bold green]Preparing data for modeling...", spinner="dots"):
        X = df[['text', 'importance']]
        y = df['category']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
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
        pipeline_tree = Tree("[bold blue]Pipeline Components")
        
        generation_node = pipeline_tree.add("[bold cyan]Feature Generation")
        generation_node.add("[green]Text Processing [dim](TF-IDF, n-grams, text features)[/dim]")
        generation_node.add("[green]Feature Interactions [dim](multiplication)[/dim]")
        
        selection_node = pipeline_tree.add("[bold cyan]Feature Selection")
        selection_node.add("[yellow]Embedded Selector [dim](Random Forest, threshold=0.001)[/dim]")
        
        console.print(pipeline_tree)
        
        # Create feature generators
        task = progress.add_task("[green]Creating feature generators...", total=1)
        
        generators = build_generation_pipeline(
            numerical=True,
            categorical=False,
            datetime=False,
            text=True,
            interactions=True,
            feature_config={
                'text': {
                    'vectorizer': 'tfidf',
                    'max_features': 200,
                    'ngram_range': (1, 2),
                    'text_columns': ['text'],
                    'advanced_features': True
                },
                'interaction': {
                    'interaction_types': ['multiplication'],
                    'max_features': 10
                },
            }
        )
        progress.update(task, advance=1)
        
        # Create pipeline
        task = progress.add_task("[green]Creating pipeline...", total=1)
        
        pipeline = FeaturePipeline(
            generation_steps=generators,
            selection_method='embedded',
            selection_params={
                'model_type': 'classification',
                'model_name': 'random_forest',
                'threshold': 0.001
            },
            target_metric='accuracy',
            max_features=50,
            verbose=1
        )
        progress.update(task, advance=1)
        
        # Fit and transform
        task = progress.add_task("[green]Fitting pipeline...", total=100)
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
    
    # Model training and evaluation
    with console.status("[bold green]Training and evaluating model...", spinner="dots"):
        # Train a model with the transformed features
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_transformed, y_train)
        
        # Evaluate on test set
        y_pred = rf.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=categories, output_dict=True)
    
    # Display model performance
    console.print("\n[bold cyan]Model Performance[/bold cyan]")
    performance_table = Table(box=box.ROUNDED)
    performance_table.add_column("Class", style="cyan")
    performance_table.add_column("Precision", style="green")
    performance_table.add_column("Recall", style="yellow")
    performance_table.add_column("F1-Score", style="magenta")
    performance_table.add_column("Support", style="blue")
    
    for category in categories:
        performance_table.add_row(
            category,
            f"{report[category]['precision']:.4f}",
            f"{report[category]['recall']:.4f}",
            f"{report[category]['f1-score']:.4f}",
            str(int(report[category]['support']))
        )
    
    # Add accuracy row
    performance_table.add_row(
        "[bold]Accuracy[/bold]",
        "",
        "",
        f"[bold]{accuracy:.4f}[/bold]",
        f"{int(report['macro avg']['support'])}"
    )
    
    console.print(performance_table)
    
    # Display selected features
    console.print("\n[bold cyan]Top Selected Text Features[/bold cyan]")
    
    # Get top text features
    features = pipeline.get_selected_features()
    text_features = [f for f in features if f.startswith('text_')][:20]  # Get top 20 text features
    
    # Split features into 2 columns for display
    chunk_size = max(1, len(text_features) // 2)
    feature_chunks = [text_features[i:i + chunk_size] for i in range(0, len(text_features), chunk_size)]
    
    # Create panels for each chunk
    panels = []
    for chunk in feature_chunks:
        feature_list = "\n".join([f"[green]• [bold cyan]{feature}[/bold cyan][/green]" for feature in chunk])
        panels.append(Panel(feature_list, expand=True))
    
    # Create Columns with all panels
    feature_columns = Columns(panels)
    console.print(feature_columns)
    
    # Get the evaluator and plot feature importances
    evaluator = pipeline.get_evaluator()
    if evaluator:
        with console.status("[bold green]Plotting feature importances...", spinner="dots"):
            plt.figure(figsize=(12, 8))
            evaluator.plot_feature_importances(top_n=20, method='aggregate')
            plt.tight_layout()
            plt.savefig('text_feature_importances.png')
        console.print("[green]Feature importances plot saved as 'text_feature_importances.png'[/green]")
    
    # Completion message
    console.print(Panel(
        "[bold green]Text Feature Engineering Example completed successfully![/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))


if __name__ == "__main__":
    main() 