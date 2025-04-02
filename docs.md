# AutoFeature Documentation

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Quick Start Guide](#quick-start-guide)
- [Feature Generation](#feature-generation)
  - [Mathematical Transformations](#mathematical-transformations)
  - [Interaction Features](#interaction-features)
  - [Aggregation Features](#aggregation-features)
  - [Time-Based Features](#time-based-features)
  - [Text-Based Features](#text-based-features)
- [Feature Selection](#feature-selection)
  - [Filter-Based Selection](#filter-based-selection)
  - [Wrapper-Based Selection](#wrapper-based-selection)
  - [Embedded Methods](#embedded-methods)
  - [Genetic Algorithm](#genetic-algorithm)
- [Feature Evaluation](#feature-evaluation)
  - [Model-Based Importance](#model-based-importance)
  - [Permutation Importance](#permutation-importance)
  - [Drop-Column Importance](#drop-column-importance)
  - [Correlation Analysis](#correlation-analysis)
- [Pipeline Integration](#pipeline-integration)
  - [Building Custom Pipelines](#building-custom-pipelines)
  - [Integration with Scikit-Learn](#integration-with-scikit-learn)
  - [Saving and Loading Pipelines](#saving-and-loading-pipelines)
- [Command-Line Interface](#command-line-interface)
- [Advanced Usage](#advanced-usage)
  - [Custom Transformers](#custom-transformers)
  - [Custom Selectors](#custom-selectors)
  - [Performance Optimization](#performance-optimization)
  - [Working with Large Datasets](#working-with-large-datasets)
- [Use Cases](#use-cases)
  - [Tabular Data](#tabular-data)
  - [Time Series Analysis](#time-series-analysis)
  - [Natural Language Processing](#natural-language-processing)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## Introduction

AutoFeature is a comprehensive automated feature engineering framework designed to enhance the performance of machine learning models by automatically generating, selecting, and evaluating features. Feature engineering is often the most crucial and time-consuming part of the machine learning workflow. AutoFeature automates this process, allowing data scientists and ML engineers to focus on model development and business insights.

**Key Benefits:**

- **Automated Feature Discovery**: Uncover complex patterns and relationships in your data that might be missed with manual feature engineering.
- **Improved Model Performance**: Generate features that can significantly boost model accuracy and generalization.
- **Time Efficiency**: Reduce the time spent on feature engineering from days to minutes.
- **Reproducibility**: Ensure consistent feature engineering processes across different datasets and projects.
- **Integration**: Seamlessly integrate with existing ML pipelines and frameworks.

AutoFeature is designed for both enterprise applications and individual projects, offering scalability, performance, and flexibility.

## Installation

### Basic Installation

Install AutoFeature using pip:

```bash
pip install autofeature
```

### Installation with Extra Dependencies

For NLP capabilities:

```bash
pip install autofeature[nlp]
```

For development purposes:

```bash
pip install autofeature[dev]
```

### From Source

Clone the repository and install:

```bash
git clone https://github.com/yourusername/autofeature.git
cd autofeature
pip install -e .
```

## Core Concepts

Automated feature engineering with AutoFeature revolves around three key components:

1. **Feature Generation**: Creating new features from existing ones using various transformations.
2. **Feature Selection**: Identifying the most relevant features for the target variable.
3. **Feature Evaluation**: Assessing the impact of features on model performance.

These components come together in a cohesive pipeline that handles the entire feature engineering workflow.

### Feature Engineering Process

1. **Data Input**: Raw data is fed into the pipeline.
2. **Feature Generation**: Multiple transformers generate new features based on the input data.
3. **Feature Selection**: Selectors identify the most relevant features for the prediction task.
4. **Feature Evaluation**: Evaluators assess the impact of selected features on model performance.
5. **Output**: Transformed dataset with optimized features.

## Quick Start Guide

### Basic Example

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from autofeature.pipeline import FeaturePipeline, build_generation_pipeline

# Load data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Add a datetime column for demonstration
X['date'] = pd.date_range(start='2020-01-01', periods=len(X))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and run the pipeline
generators = build_generation_pipeline(
    numerical=True,
    categorical=True,
    datetime=True,
    interactions=True,
    feature_config={
        'mathematical': {
            'operations': ['square', 'log', 'sqrt'],
        },
        'interaction': {
            'interaction_types': ['multiplication'],
            'max_features': 20
        }
    }
)

pipeline = FeaturePipeline(
    generation_steps=generators,
    target_metric='r2',
    max_features=15,
    verbose=1
)

# Fit and transform
X_train_transformed = pipeline.fit_transform(X_train, y_train)
X_test_transformed = pipeline.transform(X_test)

# Get statistics
stats = pipeline.get_runtime_stats()
print(f"Original features: {stats['original_features']}")
print(f"Generated features: {stats['generated_features']}")
print(f"Selected features: {stats['selected_features']}")

# View selected features
selected_features = pipeline.get_selected_features()
print("Selected features:", selected_features)
```

### Using the CLI

AutoFeature also provides a command-line interface for quick experimentation:

```bash
python -m autofeature --input data.csv --target target_column --output results_dir
```

For more advanced configurations, use a JSON config file:

```bash
python -m autofeature --input data.csv --target target_column --config config.json
```

Example config.json:
```json
{
  "generation": {
    "numerical": true,
    "categorical": true,
    "datetime": true,
    "text": false,
    "interactions": true,
    "feature_config": {
      "mathematical": {
        "operations": ["square", "log", "sqrt", "reciprocal"],
        "apply_to": "all"
      },
      "time": {
        "extract_components": true,
        "cyclical_encoding": true,
        "create_lags": true,
        "lag_values": [1, 7, 30],
        "rolling_windows": [7, 14, 30],
        "rolling_functions": ["mean", "std", "min", "max"]
      }
    }
  },
  "selection": {
    "selection_method": "embedded",
    "selection_params": {
      "model_name": "random_forest",
      "threshold": 0.001
    },
    "max_features": 50
  }
}
```

## Feature Generation

AutoFeature includes several feature generators that create new features from your existing data.

### Mathematical Transformations

The `MathematicalTransformer` applies mathematical operations to numerical features.

```python
from autofeature.feature_generation import MathematicalTransformer

# Available operations: 'square', 'cube', 'sqrt', 'log', 'abs', 'reciprocal'
transformer = MathematicalTransformer(
    operations=['square', 'log', 'sqrt'],
    apply_to='all',  # or provide a list of column names
    exclude_columns=['id', 'date']
)

X_transformed = transformer.fit_transform(X)
```

**Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `operations` | List of mathematical operations to apply | `['square', 'sqrt', 'log']` |
| `apply_to` | 'all' or list of column names | 'all' |
| `exclude_columns` | Columns to exclude from transformation | `[]` |
| `handle_special` | How to handle special cases (e.g., log of negative numbers) | 'ignore' |

### Interaction Features

The `InteractionTransformer` creates interactions between pairs of numerical features.

```python
from autofeature.feature_generation import InteractionTransformer

# Available interaction types: 'multiplication', 'division', 'addition', 'subtraction'
transformer = InteractionTransformer(
    interaction_types=['multiplication', 'division'],
    max_features=20,
    column_pairs=None  # Specify pairs or let the transformer find them
)

X_transformed = transformer.fit_transform(X)
```

**Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `interaction_types` | Types of interactions to generate | `['multiplication']` |
| `max_features` | Maximum number of interaction features to generate | 100 |
| `column_pairs` | Specific pairs of columns to interact | None (auto-detect) |
| `include_orig` | Whether to include original features in output | True |
| `random_state` | Random seed for reproducibility | 42 |

### Aggregation Features

The `AggregationTransformer` creates aggregation features for categorical variables.

```python
from autofeature.feature_generation import AggregationTransformer

transformer = AggregationTransformer(
    groupby_columns=['category', 'region'],
    agg_columns=None,  # Auto-detect numerical columns
    agg_functions=['mean', 'min', 'max', 'std']
)

X_transformed = transformer.fit_transform(X)
```

**Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `groupby_columns` | Columns to group by | Required |
| `agg_columns` | Columns to aggregate | None (auto-detect) |
| `agg_functions` | Aggregation functions to apply | `['mean', 'median', 'min', 'max', 'std']` |
| `include_orig` | Whether to include original features | True |

### Time-Based Features

The `TimeBasedTransformer` extracts features from datetime columns.

```python
from autofeature.feature_generation import TimeBasedTransformer

transformer = TimeBasedTransformer(
    datetime_columns=['date'],
    extract_components=True,  # Extract year, month, day, etc.
    cyclical_encoding=True,   # Encode cyclical features (hour, day, month)
    create_lags=True,         # Create lagged features
    lag_values=[1, 7, 30],    # Lag periods
    rolling_windows=[7, 30],  # Rolling window sizes
    rolling_functions=['mean', 'std']  # Rolling functions
)

X_transformed = transformer.fit_transform(X)
```

**Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `datetime_columns` | Datetime columns to transform | Required |
| `extract_components` | Extract date components | True |
| `cyclical_encoding` | Use sine/cosine encoding for cyclical features | True |
| `create_lags` | Create lagged features | False |
| `lag_values` | Periods for lagged features | `[1, 7, 30]` |
| `rolling_windows` | Sizes for rolling windows | `[7, 30]` |
| `rolling_functions` | Functions for rolling windows | `['mean', 'std', 'min', 'max']` |

### Text-Based Features

The `TextBasedTransformer` extracts features from text columns.

```python
from autofeature.feature_generation import TextBasedTransformer

transformer = TextBasedTransformer(
    text_columns=['description', 'comments'],
    vectorizer='tfidf',       # or 'count'
    max_features=100,         # Max features per text column
    ngram_range=(1, 2),       # Use unigrams and bigrams
    advanced_features=True    # Extract additional text metrics
)

X_transformed = transformer.fit_transform(X)
```

**Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `text_columns` | Text columns to transform | Required |
| `vectorizer` | Type of vectorizer ('tfidf' or 'count') | 'tfidf' |
| `max_features` | Maximum number of features to extract per column | 100 |
| `ngram_range` | Range of n-grams to extract | (1, 1) |
| `advanced_features` | Extract additional text metrics | False |
| `stop_words` | Stop words to remove | 'english' |

## Feature Selection

After generating features, it's important to select the most relevant ones to improve model performance and reduce dimensionality.

### Filter-Based Selection

Filter methods select features based on statistical measures without using a machine learning model.

```python
from autofeature.feature_selection import FilterSelector

selector = FilterSelector(
    method='variance',       # or 'correlation', 'chi2', 'mutual_info'
    threshold=0.01,          # Threshold for feature selection
    k=10,                    # Top k features (for ranking methods)
    target_type='auto'       # or 'regression', 'classification'
)

X_selected = selector.fit_transform(X, y)
```

**Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `method` | Selection method | 'variance' |
| `threshold` | Threshold for selection | 0.01 |
| `k` | Number of top features to select | None |
| `target_type` | Type of target variable | 'auto' |

### Wrapper-Based Selection

Wrapper methods use a machine learning model to evaluate feature subsets.

```python
from autofeature.feature_selection import WrapperSelector

selector = WrapperSelector(
    model_type='regression',      # or 'classification'
    model_name='random_forest',   # Model to use for evaluation
    direction='forward',          # 'forward', 'backward', or 'bidirectional'
    scoring='r2',                 # Metric for evaluation
    cv=5                          # Cross-validation folds
)

X_selected = selector.fit_transform(X, y)
```

**Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_type` | Type of problem | 'auto' |
| `model_name` | Model to use | 'random_forest' |
| `direction` | Search direction | 'forward' |
| `scoring` | Evaluation metric | None (depends on model_type) |
| `cv` | Cross-validation folds | 5 |
| `n_jobs` | Number of parallel jobs | -1 |

### Embedded Methods

Embedded methods use models that have built-in feature selection.

```python
from autofeature.feature_selection import EmbeddedSelector

selector = EmbeddedSelector(
    model_type='classification',    # or 'regression'
    model_name='random_forest',     # or 'lasso', 'ridge', 'elasticnet'
    threshold=0.01,                 # Importance threshold
    max_features=20                 # Maximum features to select
)

X_selected = selector.fit_transform(X, y)
```

**Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_type` | Type of problem | 'auto' |
| `model_name` | Model to use | 'random_forest' |
| `threshold` | Importance threshold | 0.01 |
| `max_features` | Maximum features to select | None |
| `hyperparams` | Hyperparameters for the model | None |

### Genetic Algorithm

The genetic algorithm approach evolves feature subsets to maximize model performance.

```python
from autofeature.feature_selection import GeneticSelector

selector = GeneticSelector(
    model_type='regression',        # or 'classification'
    cv=5,                           # Cross-validation folds
    scoring='neg_mean_squared_error',
    n_generations=10,               # Number of generations
    population_size=50,             # Size of population
    mutation_rate=0.1,              # Mutation probability
    crossover_rate=0.5,             # Crossover probability
    tournament_size=3,              # Tournament selection size
    min_features=5                  # Minimum features to select
)

X_selected = selector.fit_transform(X, y)
```

**Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_type` | Type of problem | 'auto' |
| `cv` | Cross-validation folds | 5 |
| `scoring` | Evaluation metric | None (depends on model_type) |
| `n_generations` | Number of generations | 10 |
| `population_size` | Size of population | 50 |
| `mutation_rate` | Mutation probability | 0.1 |
| `crossover_rate` | Crossover probability | 0.5 |
| `tournament_size` | Tournament selection size | 3 |
| `elite_size` | Number of elites to keep | 1 |
| `min_features` | Minimum features to select | 1 |

## Feature Evaluation

AutoFeature includes tools to evaluate the impact of features on model performance.

```python
from autofeature.feature_evaluation import FeatureEvaluator

evaluator = FeatureEvaluator(
    model_type='regression',      # or 'classification'
    model_instance=None,          # Provide a model or use default
    cv=5,                         # Cross-validation folds
    metrics=['r2', 'neg_mean_squared_error'],
    verbose=1
)

evaluator.fit(X, y)

# Get feature importances
importances = evaluator.feature_importances_

# Plot feature importances
evaluator.plot_feature_importances(top_n=10, method='aggregate')

# Plot correlation matrix
evaluator.plot_correlation_matrix(X)
```

**Configuration Options:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_type` | Type of problem | 'auto' |
| `model_instance` | Custom model instance | None |
| `cv` | Cross-validation strategy | 5 |
| `metrics` | Evaluation metrics | None (depends on model_type) |
| `verbose` | Verbosity level | 1 |

## Pipeline Integration

The `FeaturePipeline` class brings together feature generation, selection, and evaluation into a single workflow.

### Building Custom Pipelines

```python
from autofeature.pipeline import FeaturePipeline
from autofeature.feature_generation import MathematicalTransformer, InteractionTransformer
from autofeature.feature_selection import EmbeddedSelector

# Create individual components
math_transformer = MathematicalTransformer(operations=['square', 'log'])
interaction_transformer = InteractionTransformer(interaction_types=['multiplication'])
selector = EmbeddedSelector(model_name='random_forest')

# Create pipeline
pipeline = FeaturePipeline(
    generation_steps=[math_transformer, interaction_transformer],
    selection_method=selector,
    target_metric='r2',
    max_features=20,
    verbose=1
)

# Fit and transform
X_transformed = pipeline.fit_transform(X, y)
```

### Using build_generation_pipeline Helper

The `build_generation_pipeline` function creates a generation pipeline with appropriate defaults.

```python
from autofeature.pipeline import build_generation_pipeline, FeaturePipeline

# Create generators with defaults
generators = build_generation_pipeline(
    numerical=True,
    categorical=True,
    datetime=True,
    text=False,
    interactions=True
)

# Or customize configurations
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
        'time': {
            'datetime_columns': ['date'],
            'extract_components': True,
            'cyclical_encoding': True
        }
    }
)

# Create pipeline
pipeline = FeaturePipeline(
    generation_steps=generators,
    selection_method='embedded',  # Can use string shorthand for common methods
    selection_params={
        'model_type': 'regression',
        'model_name': 'random_forest',
        'threshold': 0.01
    },
    target_metric='r2'
)
```

### Integration with Scikit-Learn

AutoFeature components are compatible with scikit-learn pipelines.

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from autofeature.feature_generation import MathematicalTransformer
from autofeature.feature_selection import FilterSelector

# Create a scikit-learn pipeline
pipeline = Pipeline([
    ('math_features', MathematicalTransformer(operations=['square', 'log'])),
    ('feature_selection', FilterSelector(method='variance', threshold=0.01)),
    ('model', RandomForestRegressor())
])

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### Saving and Loading Pipelines

Save your feature engineering pipeline for reuse:

```python
# Save the pipeline
pipeline.save_pipeline('pipeline.pkl')

# Later, load the pipeline
from autofeature.utils.io import load_pipeline
pipeline = load_pipeline('pipeline.pkl')

# Transform new data
X_new_transformed = pipeline.transform(X_new)
```

## Command-Line Interface

AutoFeature provides a command-line interface for easy usage:

```bash
# Basic usage
python -m autofeature --input data.csv --target target_column

# Specify feature generation options
python -m autofeature --input data.csv --target target_column \
    --numerical --datetime --interactions \
    --datetime-cols date purchase_date

# Specify feature selection method
python -m autofeature --input data.csv --target target_column \
    --selection-method wrapper \
    --model random_forest \
    --max-features 20

# Save results
python -m autofeature --input data.csv --target target_column \
    --output results_dir \
    --save-pipeline --save-transformed
```

**CLI Options:**

| Option | Description |
|--------|-------------|
| `--input` | Path to input data file (CSV, Excel, or Parquet) |
| `--output` | Directory to save output files |
| `--target` | Name of the target column |
| `--task` | Type of machine learning task ('auto', 'regression', 'classification') |
| `--numerical` | Enable numerical feature transformations |
| `--categorical` | Enable categorical feature transformations |
| `--datetime` | Enable datetime feature transformations |
| `--text` | Enable text feature transformations |
| `--interactions` | Enable feature interactions |
| `--selection-method` | Feature selection method |
| `--max-features` | Maximum number of features to select |
| `--model` | Model to use for feature selection |
| `--config` | Path to JSON configuration file |

## Advanced Usage

### Custom Transformers

Create custom feature generators by extending the base transformer class:

```python
from autofeature.feature_generation import BaseTransformer
import pandas as pd
import numpy as np

class CustomTransformer(BaseTransformer):
    def __init__(self, custom_param=1):
        super().__init__()
        self.custom_param = custom_param
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """Fit the transformer (if needed)."""
        self._validate_input(X)
        # Perform fitting operations
        self.is_fitted = True
        return self
        
    def transform(self, X):
        """Transform the data with custom logic."""
        if not self.is_fitted:
            raise ValueError("Transformer is not fitted.")
            
        self._validate_input(X)
        result = X.copy()
        
        # Apply custom transformations
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                result[f"{col}_custom"] = np.log(X[col] + 1) * self.custom_param
                
        return result
```

### Custom Selectors

Create custom feature selectors by extending the base selector class:

```python
from autofeature.feature_selection import BaseSelector
import pandas as pd
import numpy as np

class CustomSelector(BaseSelector):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        
    def fit(self, X, y=None):
        """Fit the selector."""
        self._validate_input(X)
        
        # Calculate custom importance scores
        self.feature_importances_ = {}
        for col in X.columns:
            # Example: use correlation with target for importance
            if y is not None and pd.api.types.is_numeric_dtype(X[col]):
                corr = np.abs(np.corrcoef(X[col], y)[0, 1])
                self.feature_importances_[col] = corr
            else:
                self.feature_importances_[col] = 0.0
        
        # Select features based on threshold
        self.selected_features_ = [
            col for col, importance in self.feature_importances_.items()
            if importance >= self.threshold
        ]
        
        return self
        
    def transform(self, X):
        """Select features based on fitted selector."""
        if not hasattr(self, 'selected_features_'):
            raise ValueError("Selector is not fitted.")
            
        self._validate_input(X)
        return X[self.selected_features_]
```

### Performance Optimization

For large datasets, consider these performance optimization strategies:

```python
# 1. Limit the number of generated features
generators = build_generation_pipeline(
    numerical=True,
    feature_config={
        'mathematical': {
            'operations': ['square', 'log'],  # Limit operations
            'apply_to': ['col1', 'col2']      # Apply only to specific columns
        },
        'interaction': {
            'max_features': 20                # Limit interaction features
        }
    }
)

# 2. Use embedded feature selection (faster than wrapper methods)
pipeline = FeaturePipeline(
    generation_steps=generators,
    selection_method='embedded',
    selection_params={
        'model_name': 'random_forest',
        'threshold': 0.01,
        'n_jobs': -1  # Parallelize
    }
)

# 3. Sample your data during development
X_sample = X.sample(10000, random_state=42)
y_sample = y[X_sample.index]

# Develop your pipeline on the sample
pipeline.fit(X_sample, y_sample)

# 4. Use incremental feature generation
from autofeature.feature_generation import MathematicalTransformer

# Process one transformer at a time
math_features = MathematicalTransformer().fit_transform(X)
```

### Working with Large Datasets

For datasets that don't fit in memory:

```python
import dask.dataframe as dd
from autofeature.pipeline import FeaturePipeline

# Load data with Dask
X_dask = dd.read_csv('large_data.csv')

# Process in chunks
chunk_size = 10000
n_chunks = (len(X_dask) // chunk_size) + 1

pipeline = FeaturePipeline(
    generation_steps=generators,
    selection_method='filter',  # Use filter methods for large data
    selection_params={'method': 'variance'}
)

# Process each chunk
X_transformed_chunks = []
for i in range(n_chunks):
    chunk = X_dask.iloc[i*chunk_size:(i+1)*chunk_size].compute()
    X_transformed_chunk = pipeline.transform(chunk)
    X_transformed_chunks.append(X_transformed_chunk)

# Combine results
X_transformed = pd.concat(X_transformed_chunks)
```

## Use Cases

### Tabular Data

```python
from autofeature.pipeline import FeaturePipeline, build_generation_pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
generators = build_generation_pipeline(numerical=True, interactions=True)
pipeline = FeaturePipeline(
    generation_steps=generators,
    selection_method='embedded',
    selection_params={'model_name': 'random_forest'},
    target_metric='r2'
)

# Transform data
X_train_transformed = pipeline.fit_transform(X_train, y_train)
X_test_transformed = pipeline.transform(X_test)

# Train model with original features
model_original = RandomForestRegressor(random_state=42)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
r2_original = r2_score(y_test, y_pred_original)

# Train model with transformed features
model_transformed = RandomForestRegressor(random_state=42)
model_transformed.fit(X_train_transformed, y_train)
y_pred_transformed = model_transformed.predict(X_test_transformed)
r2_transformed = r2_score(y_test, y_pred_transformed)

print(f"R² with original features: {r2_original:.4f}")
print(f"R² with transformed features: {r2_transformed:.4f}")
print(f"Improvement: {(r2_transformed - r2_original) / r2_original * 100:.2f}%")
```

### Time Series Analysis

```python
from autofeature.pipeline import FeaturePipeline, build_generation_pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Load time series data (example)
df = pd.read_csv('time_series_data.csv', parse_dates=['date'])
X = df.drop('target', axis=1)
y = df['target']

# Create time-based split
tscv = TimeSeriesSplit(n_splits=5)
train_idx, test_idx = list(tscv.split(X))[-1]  # Get last split
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Build pipeline for time series
generators = build_generation_pipeline(
    numerical=True,
    datetime=True,
    feature_config={
        'time': {
            'datetime_columns': ['date'],
            'extract_components': True,
            'cyclical_encoding': True,
            'create_lags': True,
            'lag_values': [1, 7, 14, 30],
            'rolling_windows': [7, 14, 30],
            'rolling_functions': ['mean', 'std', 'min', 'max']
        }
    }
)

pipeline = FeaturePipeline(
    generation_steps=generators,
    selection_method='wrapper',
    selection_params={
        'model_type': 'regression',
        'model_name': 'random_forest',
        'scoring': 'neg_mean_squared_error',
        'cv': TimeSeriesSplit(n_splits=3)
    },
    target_metric='neg_mean_squared_error'
)

# Transform data
X_train_transformed = pipeline.fit_transform(X_train, y_train)
X_test_transformed = pipeline.transform(X_test)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_transformed, y_train)
y_pred = model.predict(X_test_transformed)

# Evaluate
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# View important features
print("Top features:")
for feature, importance in zip(
    pipeline.get_selected_features(),
    model.feature_importances_
):
    print(f"{feature}: {importance:.4f}")
```

### Natural Language Processing

```python
from autofeature.pipeline import FeaturePipeline, build_generation_pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load text data
categories = ['alt.atheism', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories)
df = pd.DataFrame({
    'text': newsgroups.data,
    'category': newsgroups.target
})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df[['text']], df['category'], test_size=0.3, random_state=42
)

# Build pipeline for text data
generators = build_generation_pipeline(
    text=True,
    feature_config={
        'text': {
            'vectorizer': 'tfidf',
            'max_features': 1000,
            'ngram_range': (1, 2),
            'text_columns': ['text'],
            'advanced_features': True
        }
    }
)

pipeline = FeaturePipeline(
    generation_steps=generators,
    selection_method='embedded',
    selection_params={
        'model_type': 'classification',
        'model_name': 'random_forest',
        'threshold': 0.001
    },
    target_metric='accuracy',
    max_features=200
)

# Transform data
X_train_transformed = pipeline.fit_transform(X_train, y_train)
X_test_transformed = pipeline.transform(X_test)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train_transformed, y_train)
y_pred = model.predict(X_test_transformed)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# View top text features
text_features = [f for f in pipeline.get_selected_features() if f.startswith('text_')]
print(f"Top 10 text features (out of {len(text_features)}):")
for feature in text_features[:10]:
    print(f"- {feature}")
```

## API Reference

For detailed API documentation, please refer to the [API Reference](api_reference.md) document.

## Troubleshooting

### Common Issues

**Memory Errors with Large Datasets**

```python
# Solution 1: Reduce feature generation
generators = build_generation_pipeline(
    numerical=True,
    feature_config={
        'mathematical': {'operations': ['square']},  # Minimal transformations
        'interaction': {'max_features': 10}          # Few interaction features
    }
)

# Solution 2: Process in chunks
chunk_size = 10000
for i in range(0, len(X), chunk_size):
    X_chunk = X.iloc[i:i+chunk_size]
    # Process chunk...
```

**Performance Issues with Wrapper Methods**

```python
# Solution: Use embedded selection instead
pipeline = FeaturePipeline(
    generation_steps=generators,
    selection_method='embedded',          # Faster than wrapper
    selection_params={
        'model_name': 'random_forest',
        'n_jobs': -1                      # Parallelize
    }
)
```

**Issues with Text Features**

```python
# Solution: Install NLP dependencies
# pip install autofeature[nlp]

# And limit text features
generators = build_generation_pipeline(
    text=True,
    feature_config={
        'text': {
            'max_features': 100,          # Limit features
            'ngram_range': (1, 1)         # Only unigrams
        }
    }
)
```

For more troubleshooting and tips, see the [FAQ](faq.md) document. 