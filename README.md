# AutoFeature: Automated Feature Engineering Framework

AutoFeature is a comprehensive Python framework for automated feature engineering in machine learning projects. It streamlines the feature engineering process by automatically generating, selecting, and evaluating features to improve model performance.

## Key Features

- **Automated Feature Generation**: Create new features through various transformations:
  - Mathematical transformations (square, log, sqrt, etc.)
  - Feature interactions (multiplication, division, etc.)
  - Aggregation features (mean, median, etc. across groups)
  - Time-based features (extract components and cycles from datetime)
  - Text-based features (extract statistics and patterns from text)

- **Intelligent Feature Selection**: Select the most valuable features using:
  - Filter methods (statistical measures)
  - Wrapper methods (model performance-based)
  - Embedded methods (model-based selection)
  - Genetic algorithms (evolutionary approach)

- **Feature Evaluation**: Measure the impact of features on model performance:
  - Model-based importance
  - Permutation importance
  - Drop-column importance
  - Feature correlation analysis

- **Integrated Pipeline**: Seamlessly combine generation, selection, and evaluation:
  - Preprocessing and handling of different data types
  - Configurable pipeline for customization
  - Performance tracking and runtime statistics
  - Easy integration with scikit-learn workflows

## Installation

```bash
# Install from PyPI
pip install autofeature

# Install with additional NLP dependencies
pip install autofeature[nlp]

# Install development dependencies
pip install autofeature[dev]
```

## Quick Start

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from autofeature.pipeline import FeaturePipeline, build_generation_pipeline

# Load your data
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up feature generators
generators = build_generation_pipeline(
    numerical=True,
    categorical=True,
    datetime=True,
    text=False,
    interactions=True
)

# Create and run the pipeline
pipeline = FeaturePipeline(
    generation_steps=generators,
    target_metric='r2',
    max_features=20,
    verbose=1
)

# Fit and transform
X_train_transformed = pipeline.fit_transform(X_train, y_train)
X_test_transformed = pipeline.transform(X_test)

# Get selected features
selected_features = pipeline.get_selected_features()
print(f"Selected features: {selected_features}")

# Get runtime statistics
stats = pipeline.get_runtime_stats()
print(f"Generated {stats['generated_features']} features in {stats['generation_time']:.2f} seconds")
print(f"Selected {stats['selected_features']} features in {stats['selection_time']:.2f} seconds")
```

## Customizing Feature Generation

You can customize the feature generation process by configuring each transformer:

```python
from autofeature.feature_generation import (
    MathematicalTransformer, 
    InteractionTransformer, 
    TimeBasedTransformer
)

# Create custom transformers
math_transformer = MathematicalTransformer(
    operations=['square', 'log', 'sqrt', 'reciprocal'],
    exclude_columns=['id', 'name']
)

interaction_transformer = InteractionTransformer(
    interaction_types=['multiplication', 'division', 'addition'],
    max_combinations=2
)

time_transformer = TimeBasedTransformer(
    components=['year', 'month', 'day_of_week'],
    cyclical=True
)

# Create pipeline with custom transformers
pipeline = FeaturePipeline(
    generation_steps=[math_transformer, interaction_transformer, time_transformer],
    selection_method='genetic',
    max_features=30
)
```

## Feature Selection Methods

AutoFeature provides several feature selection methods:

```python
from autofeature.feature_selection import (
    FilterSelector,
    WrapperSelector,
    EmbeddedSelector,
    GeneticSelector
)

# Filter-based selection
filter_selector = FilterSelector(
    method='f_regression',
    k=20
)

# Wrapper-based selection
wrapper_selector = WrapperSelector(
    method='forward',
    model_type='regression',
    cv=5
)

# Embedded selection
embedded_selector = EmbeddedSelector(
    model_name='random_forest',
    threshold='mean'
)

# Genetic algorithm selection
genetic_selector = GeneticSelector(
    n_generations=10,
    population_size=50,
    mutation_rate=0.1
)
```

## Feature Evaluation

Evaluate the impact of features on model performance:

```python
from autofeature.feature_evaluation import FeatureEvaluator
import matplotlib.pyplot as plt

# Create evaluator
evaluator = pipeline.get_evaluator()

# Get feature evaluations
feature_evals = evaluator.evaluate_features(top_n=20)
print(feature_evals)

# Plot feature importances
fig = evaluator.plot_feature_importances(
    top_n=15,
    method='aggregate'
)
plt.show()

# Plot correlation matrix
fig = evaluator.plot_correlation_matrix(X_train_transformed)
plt.show()
```

## Advanced Usage

### Custom Pipeline Integration

Integrate with scikit-learn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Create feature engineering pipeline
feature_pipeline = FeaturePipeline(
    generation_steps=build_generation_pipeline(),
    max_features=20
)

# Create sklearn pipeline
full_pipeline = Pipeline([
    ('features', feature_pipeline),
    ('model', RandomForestRegressor())
])

# Fit and predict
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

### Saving and Loading

Save your pipeline configuration:

```python
# Save pipeline configuration
pipeline.save_pipeline('pipeline_config.json')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 