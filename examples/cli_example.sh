#!/bin/bash
# Example CLI usage for AutoFeature framework

# Basic usage with default settings
echo "Basic usage example"
echo "-------------------"
echo "python -m autofeature --input data/sample.csv --target price"
echo

# Using a configuration file
echo "Using configuration file"
echo "-----------------------"
echo "python -m autofeature --input data/sample.csv --target price --config examples/config.json"
echo

# Specifying feature generation options
echo "Customizing feature generation"
echo "-----------------------------"
echo "python -m autofeature --input data/sample.csv --target price \\"
echo "    --numerical --datetime --interactions \\"
echo "    --datetime-cols date purchase_date \\"
echo "    --verbose 2"
echo

# Specifying feature selection method
echo "Customizing feature selection"
echo "----------------------------"
echo "python -m autofeature --input data/sample.csv --target price \\"
echo "    --selection-method wrapper \\"
echo "    --model random_forest \\"
echo "    --cv 5 \\"
echo "    --max-features 20"
echo

# Time series example
echo "Time series example"
echo "------------------"
echo "python -m autofeature --input data/timeseries.csv --target sales \\"
echo "    --task regression \\"
echo "    --datetime \\"
echo "    --datetime-cols date \\"
echo "    --selection-method wrapper \\"
echo "    --scoring neg_mean_absolute_error"
echo

# Text processing example
echo "Text processing example"
echo "----------------------"
echo "python -m autofeature --input data/reviews.csv --target sentiment \\"
echo "    --task classification \\"
echo "    --text \\"
echo "    --text-cols review_text \\"
echo "    --selection-method embedded \\"
echo "    --scoring accuracy"
echo

# Saving and loading example
echo "Saving and loading example"
echo "-------------------------"
echo "# Generate and save features"
echo "python -m autofeature --input data/sample.csv --target price \\"
echo "    --output results/feature_engineering \\"
echo "    --save-pipeline --save-transformed"
echo
echo "# Later, load the pipeline for new data"
echo "python -c \"from autofeature.utils.io import load_pipeline; \\"
echo "           pipeline = load_pipeline('results/feature_engineering/pipeline_YYYYMMDD_HHMMSS.pkl'); \\"
echo "           # Now use pipeline.transform() on new data\"" 