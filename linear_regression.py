"""
Linear Regression Model for House Price Prediction
Capstone Project - Classical ML Algorithm #1
"""

import os
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# Custom Transformers
# ============================================================================

class Standardization(BaseEstimator, TransformerMixin):
    """Standardize features by removing mean and scaling to unit variance"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: pd.DataFrame, y=None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return (X - self.mean) / self.std

class AddBias(BaseEstimator, TransformerMixin):
    """Add bias term (column of 1s) to feature matrix"""
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        X.insert(0, 'bias', 1)
        return X

# ============================================================================
# Data Preprocessing Pipeline
# ============================================================================

def load_data(filepath='train.csv') -> pd.DataFrame:
    """Load the Ames Housing dataset"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data before splitting"""
    df = df.copy()
    
    # Separate target
    if 'SalePrice' in df.columns:
        target = df['SalePrice']
        features = df.drop(['SalePrice', 'Id'], axis=1, errors='ignore')
    else:
        raise ValueError("SalePrice column not found in dataset")
    
    # Handle missing values for categorical features
    cat_features = features.select_dtypes(include=['object']).columns
    for col in cat_features:
        features[col] = features[col].fillna('None')
    
    # Handle missing values for numerical features
    num_features = features.select_dtypes(include=[np.number]).columns
    for col in num_features:
        features[col] = features[col].fillna(features[col].median())
    
    # One-hot encode categorical variables
    features_encoded = pd.get_dummies(features, drop_first=True)
    
    # Recombine with target
    df_clean = features_encoded.copy()
    df_clean['SalePrice'] = target
    
    print(f"After preprocessing: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    print(f"Features: {df_clean.shape[1] - 1}, Target: SalePrice")
    
    return df_clean

def feature_label_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and labels"""
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice'].copy()
    return X, y

def train_valid_test_split(X: pd.DataFrame, y: pd.Series) -> Tuple:
    """Split data into train, validation, and test sets"""
    # First split: separate test set (20%)
    X_temp, X_tst, y_temp, y_tst = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Second split: separate validation set (20% of remaining = 16% of total)
    X_trn, X_vld, y_trn, y_vld = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    # Reset indices
    for data in [X_trn, y_trn, X_vld, y_vld, X_tst, y_tst]:
        data.reset_index(inplace=True, drop=True)
    
    print(f"\nData split:")
    print(f"  Training:   {X_trn.shape[0]} samples ({X_trn.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validation: {X_vld.shape[0]} samples ({X_vld.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test:       {X_tst.shape[0]} samples ({X_tst.shape[0]/len(X)*100:.1f}%)")
    
    return X_trn, y_trn, X_vld, y_vld, X_tst, y_tst

def apply_feature_pipeline(X_trn: pd.DataFrame, 
                           X_vld: pd.DataFrame, 
                           X_tst: pd.DataFrame) -> Tuple:
    """Apply standardization and bias term to features"""
    stages = [
        ('scaler', Standardization()),
        ('bias', AddBias())
    ]
    
    pipeline = Pipeline(stages)
    
    # Fit on training data only
    X_trn_scaled = pipeline.fit_transform(X_trn)
    X_vld_scaled = pipeline.transform(X_vld)
    X_tst_scaled = pipeline.transform(X_tst)
    
    print(f"\nFeature engineering applied:")
    print(f"  Standardization: μ=0, σ=1")
    print(f"  Bias term added")
    print(f"  Final feature count: {X_trn_scaled.shape[1]}")
    
    return X_trn_scaled, X_vld_scaled, X_tst_scaled

# ============================================================================
# Model Training and Evaluation
# ============================================================================

def train_model(X_trn: pd.DataFrame, y_trn: pd.Series) -> LinearRegression:
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X_trn, y_trn)
    
    print("\nModel trained: Linear Regression")
    print(f"  Coefficients learned: {X_trn.shape[1]}")
    
    return model

def evaluate_model(model: LinearRegression, 
                   X: pd.DataFrame, 
                   y: pd.Series, 
                   dataset_name: str) -> dict:
    """Evaluate model performance"""
    y_pred = model.predict(X)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  R²:   {r2:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def cross_validate_model(model: LinearRegression, 
                         X: pd.DataFrame, 
                         y: pd.Series, 
                         cv: int = 5):
    """Perform cross-validation"""
    cv_scores = cross_val_score(
        model, X, y, cv=cv, 
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    cv_rmse = np.sqrt(-cv_scores.mean())
    cv_std = np.sqrt(cv_scores.std())
    
    print(f"\nCross-Validation Results ({cv}-fold):")
    print(f"  Mean RMSE: ${cv_rmse:,.2f}")
    print(f"  Std Dev:   ${cv_std:,.2f}")
    
    return cv_rmse

# ============================================================================
# Visualization
# ============================================================================

def plot_results(y_true: pd.Series, 
                 y_pred: np.ndarray, 
                 dataset_name: str,
                 save_path: str = 'linear_regression_results.png'):
    """Create visualization of model performance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price ($)', fontsize=12)
    axes[0].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[0].set_title(f'Linear Regression: Actual vs Predicted ({dataset_name})', 
                     fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Price ($)', fontsize=12)
    axes[1].set_ylabel('Residuals ($)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {save_path}")
    plt.show()

def save_metrics(train_metrics: dict, 
                 val_metrics: dict, 
                 test_metrics: dict,
                 cv_rmse: float,
                 filepath: str = 'linear_regression_metrics.csv'):
    """Save metrics to CSV file"""
    metrics_df = pd.DataFrame({
        'train_rmse': [train_metrics['rmse']],
        'train_mae': [train_metrics['mae']],
        'train_r2': [train_metrics['r2']],
        'val_rmse': [val_metrics['rmse']],
        'val_mae': [val_metrics['mae']],
        'val_r2': [val_metrics['r2']],
        'test_rmse': [test_metrics['rmse']],
        'test_mae': [test_metrics['mae']],
        'test_r2': [test_metrics['r2']],
        'cv_rmse': [cv_rmse]
    })
    
    metrics_df.to_csv(filepath, index=False)
    print(f"Metrics saved: {filepath}")

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main execution pipeline"""
    print("="*70)
    print("LINEAR REGRESSION - HOUSE PRICE PREDICTION")
    print("="*70)
    
    # Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    df = load_data('train.csv')
    df_clean = preprocess_data(df)
    
    # Split features and labels
    print("\n[2/6] Splitting features and labels...")
    X, y = feature_label_split(df_clean)
    
    # Split into train/validation/test
    print("\n[3/6] Splitting into train/validation/test sets...")
    X_trn, y_trn, X_vld, y_vld, X_tst, y_tst = train_valid_test_split(X, y)
    
    # Apply feature engineering pipeline
    print("\n[4/6] Applying feature engineering...")
    X_trn_scaled, X_vld_scaled, X_tst_scaled = apply_feature_pipeline(
        X_trn, X_vld, X_tst
    )
    
    # Train model
    print("\n[5/6] Training model...")
    model = train_model(X_trn_scaled, y_trn)
    
    # Evaluate on all sets
    print("\n[6/6] Evaluating model...")
    train_metrics = evaluate_model(model, X_trn_scaled, y_trn, "Training")
    val_metrics = evaluate_model(model, X_vld_scaled, y_vld, "Validation")
    test_metrics = evaluate_model(model, X_tst_scaled, y_tst, "Test")
    
    # Cross-validation
    cv_rmse = cross_validate_model(model, X_trn_scaled, y_trn)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    plot_results(y_tst, test_metrics['predictions'], 'Test Set')
    save_metrics(train_metrics, val_metrics, test_metrics, cv_rmse)
    
    print("\n" + "="*70)
    print("LINEAR REGRESSION COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()