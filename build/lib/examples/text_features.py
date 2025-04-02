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

from autofeature.pipeline import FeaturePipeline, build_generation_pipeline


def main():
    """Run the text feature engineering example."""
    print("AutoFeature Text Feature Engineering Example")
    print("===========================================")
    
    # Load a subset of the 20 newsgroups dataset (3 categories)
    print("\nLoading 20 newsgroups dataset (subset)...")
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
    
    # Examine data
    print(f"\nDataset shape: {df.shape}")
    print("\nSample data:")
    print(df[['category_name', 'importance', 'text']].head(2))
    
    # Split features and target
    X = df[['text', 'importance']]
    y = df['category']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create feature generators with text features enabled
    print("\nCreating feature generators with text processing...")
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
    
    # Create pipeline
    print("\nCreating and running feature pipeline...")
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
    
    # Train a model with the transformed features
    print("\nTraining a model with transformed features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_transformed, y_train)
    
    # Evaluate on test set
    y_pred = rf.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"\nModel accuracy with transformed features: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))
    
    # Get the evaluator and plot feature importances
    evaluator = pipeline.get_evaluator()
    if evaluator:
        print("\nPlotting feature importances...")
        plt.figure(figsize=(12, 8))
        evaluator.plot_feature_importances(top_n=20, method='aggregate')
        plt.tight_layout()
        plt.savefig('text_feature_importances.png')
        print("Feature importances plot saved as 'text_feature_importances.png'")
    
    # Print top selected text features
    print("\nTop selected text features (sample):")
    features = pipeline.get_selected_features()
    text_features = [f for f in features if f.startswith('text_')]
    for feature in text_features[:10]:  # Show top 10 text features
        print(f"  - {feature}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 