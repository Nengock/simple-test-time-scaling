# Test-Time Scaling Demo

A Jupyter notebook-based project demonstrating the implementation and evaluation of test-time scaling strategies on a classification dataset.

## Project Overview

This project explores various test-time scaling techniques for machine learning models, comparing their effectiveness and impact on model performance. The project includes data preparation, model training, test-time scaling experiments, and comprehensive evaluation metrics.

## Project Structure

```
├── data/
│   ├── raw/                  # Original unprocessed dataset files
│   └── processed/            # Cleaned and scaled versions of datasets
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Data loading, cleaning, and exploration
│   ├── 02_model_training.ipynb      # Baseline model training
│   ├── 03_test_time_scaling.ipynb   # Test-time scaling experiments
│   └── 04_evaluation.ipynb          # Performance evaluation and visualization
├── utils/
│   └── preprocessing.py      # Data processing utility functions
├── requirements.txt          # Project dependencies
└── README.md                # This file
```

## Scaling Methods

The project implements and compares the following scaling techniques:
- StandardScaler (train-time)
- QuantileTransformer
- RobustScaler
- Z-score normalization (test-only)
- MinMaxScaler

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow

1. **Data Preparation**: Run `01_data_preparation.ipynb` to prepare and explore the dataset
2. **Model Training**: Run `02_model_training.ipynb` to train the baseline model
3. **Test-Time Scaling**: Run `03_test_time_scaling.ipynb` to experiment with different scaling methods
4. **Evaluation**: Run `04_evaluation.ipynb` to analyze and compare results

## Best Practices

- Scale training data independently of test data
- Avoid data leakage by not using test data statistics for scaling
- Utilize utility modules for code reusability
- Store intermediate outputs for efficiency
- Document analysis steps thoroughly

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter