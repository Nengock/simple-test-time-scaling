import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler, MinMaxScaler
from typing import Tuple, Optional, Union
import pandas as pd

class ScalingManager:
    """Manages different scaling strategies for training and test time."""
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'quantile': QuantileTransformer(output_distribution='normal'),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        self.fitted_scalers = {}

    def fit_scaler(self, X: np.ndarray, scaler_name: str) -> None:
        """Fit a scaler on training data."""
        if scaler_name not in self.scalers:
            raise ValueError(f"Unknown scaler: {scaler_name}")
        self.fitted_scalers[scaler_name] = self.scalers[scaler_name].fit(X)

    def transform(self, X: np.ndarray, scaler_name: str) -> np.ndarray:
        """Transform data using a fitted scaler."""
        if scaler_name not in self.fitted_scalers:
            raise ValueError(f"Scaler {scaler_name} not fitted yet")
        return self.fitted_scalers[scaler_name].transform(X)

    @staticmethod
    def test_time_zscore(X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization at test time (for demonstration only)."""
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def load_and_preprocess_data(
    filepath: str,
    target_column: str,
    drop_columns: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess raw data."""
    data = pd.read_csv(filepath)
    
    if drop_columns:
        data = data.drop(columns=drop_columns)
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    return X, y

def handle_missing_values(
    X: pd.DataFrame,
    strategy: str = 'mean'
) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    if strategy == 'mean':
        return X.fillna(X.mean())
    elif strategy == 'median':
        return X.fillna(X.median())
    elif strategy == 'drop':
        return X.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def save_processed_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    filepath: str
) -> None:
    """Save processed data to disk."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    # Combine features and target for saving
    data = pd.concat([X, y], axis=1)
    data.to_csv(filepath, index=False)