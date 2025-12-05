"""
Polynomial Regression Model for House Price Prediction
Capstone Project - Classical ML Algorithm #2
"""

import os
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
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
# Data Preprocessing
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
    
    target = df['SalePrice']
    features = df.drop(['SalePrice', 'Id'], axis=1, errors='ignore')
    
    # Handle missing values
    cat_features = features.select_dtypes(include=['object']).columns
    for col in cat_features:
        features[col] = features[col].fillna('None')
    
    num_features = features.select_dtypes(include=[np.number]).columns
    for col in num_features:
        features[col] = features[col].fillna(features[col].median())
    
    # One-hot encode categorical variables
    features_encoded = pd.get_dummies(features, drop_first=True)
    
    df_clean = features_encoded.copy()
    df_clean['SalePrice'] = target
    
    print(f"After preprocessing: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    
    return df_clean

def feature_label_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split features and labels"""
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice'].copy()
    return X, y

def select_important_features(X: pd.DataFrame, y: pd.Series, n_features: int = 10) -> pd.DataFrame:
    """Select top features using Random Forest importance to reduce dimensionality"""
    print(f"\nSelecting top {n_features} features for polynomial expansion...")
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importances.head(n_features)['feature'].tolist()
    
    print(f"Top {n_features} features selected:")
    for i, (feat, imp) in enumerate(zip(top_features, importances.head(n_features)['importance']), 1):
        print(f"  {i:2d}. {feat:30s}: {imp:.4f}")
    
    return X[top_features]

def train_valid_test_split(X: pd.DataFrame, y: pd.Series) -> Tuple:
    """Split data into train, validation, and test sets"""
    X_temp, X_tst, y_temp, y_tst = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_trn, X_vld, y_trn, y_vld = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    # Reset indices
    for data in [X_trn, y_trn, X_vld, y_vld, X_tst, y_tst]:
        data.reset_index(inplace=True, drop=True)
    
    print(f"\nData split:")
    print(f"  Training:   {X_trn.shape[0]} samples")
    print(f"  Validation: {X_vld.shape[0]} samples")
    print(f"  Test:       {X_tst.shape[0]} samples")
    
    return X_trn, y_trn, X_vld, y_vld, X_tst, y_tst

# ============================================================================
# Hyperparameter Tuning
# ============================================================================

def tune_hyperparameters(X_trn: pd.DataFrame, 
                         y_trn: pd.Series,
                         X_vld: pd.DataFrame,
                         y_vld: pd.Series) -> Tuple[int, float]:
    """Find best polynomial degree and regularization strength"""
    print("\nHyperparameter tuning:")
    print("-" * 60)
    
    degrees = [1, 2, 3]
    alphas = [0.1, 1.0, 10.0, 100.0]
    
    best_score = float('inf')
    best_params = {}
    results = []
    
    for degree in degrees:
        for alpha in alphas:
            # Create pipeline
            pipeline = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('scaler', Standardization()),
                ('regressor', Ridge(alpha=alpha))
            ])
            
            # Train and evaluate
            pipeline.fit(X_trn, y_trn)
            y_vld_pred = pipeline.predict(X_vld)
            val_rmse = np.sqrt(mean_squared_error(y_vld, y_vld_pred))
            
            results.append({
                'degree': degree,
                'alpha': alpha,
                'val_rmse': val_rmse
            })
            
            print(f"  Degree={degree}, Alpha={alpha:6.1f} → Val RMSE: ${val_rmse:,.2f}")
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_params = {'degree': degree, 'alpha': alpha}
    
    print("-" * 60)
    print(f"Best parameters: Degree={best_params['degree']}, Alpha={best_params['alpha']}")
    print(f"Best validation RMSE: ${best_score:,.2f}")
    
    # Save tuning results
    pd.DataFrame(results).to_csv('polynomial_regression_tuning.csv', index=False)
    
    return best_params['degree'], best_params['alpha']

# ============================================================================
# Model Training and Evaluation
# ============================================================================

def train_model(X_trn: pd.DataFrame, 
                y_trn: pd.Series,
                degree: int,
                alpha: float) -> Pipeline:
    """Train Polynomial Regression model with Ridge regularization"""
    print(f"\nTraining Polynomial Regression (degree={degree}, alpha={alpha})...")
    
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', Standardization()),
        ('bias', AddBias()),
        ('regressor', Ridge(alpha=alpha))
    ])
    
    pipeline.fit(X_trn, y_trn)
    
    # Get number of polynomial features created
    n_poly_features = pipeline.named_steps['poly'].n_output_features_
    print(f"  Polynomial features created: {n_poly_features}")
    print(f"  Regularization (Ridge): α={alpha}")
    
    return pipeline

def evaluate_model(model: Pipeline, 
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

# ============================================================================
# Visualization
# ============================================================================

def plot_results(y_true: pd.Series, 
                 y_pred: np.ndarray, 
                 degree: int,
                 save_path: str = 'polynomial_regression_results.png'):
    """Create visualization of model performance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price ($)', fontsize=12)
    axes[0].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[0].set_title(f'Polynomial Regression (Degree={degree}): Actual vs Predicted', 
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
                 degree: int,
                 alpha: float,
                 filepath: str = 'polynomial_regression_metrics.csv'):
    """Save metrics to CSV file"""
    metrics_df = pd.DataFrame({
        'degree': [degree],
        'alpha': [alpha],
        'train_rmse': [train_metrics['rmse']],
        'train_mae': [train_metrics['mae']],
        'train_r2': [train_metrics['r2']],
        'val_rmse': [val_metrics['rmse']],
        'val_mae': [val_metrics['mae']],
        'val_r2': [val_metrics['r2']],
        'test_rmse': [test_metrics['rmse']],
        'test_mae': [test_metrics['mae']],
        'test_r2': [test_metrics['r2']]
    })
    
    metrics_df.to_csv(filepath, index=False)
    print(f"Metrics saved: {filepath}")

# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Main execution pipeline"""
    print("="*70)
    print("POLYNOMIAL REGRESSION - HOUSE PRICE PREDICTION")
    print("="*70)
    
    # Load and preprocess
    print("\n[1/7] Loading and preprocessing data...")
    df = load_data('train.csv')
    df_clean = preprocess_data(df)
    
    # Split features and labels
    print("\n[2/7] Splitting features and labels...")
    X, y = feature_label_split(df_clean)
    
    # Feature selection
    print("\n[3/7] Selecting important features...")
    X_selected = select_important_features(X, y, n_features=10)
    
    # Split data
    print("\n[4/7] Splitting into train/validation/test sets...")
    X_trn, y_trn, X_vld, y_vld, X_tst, y_tst = train_valid_test_split(X_selected, y)
    
    # Hyperparameter tuning
    print("\n[5/7] Tuning hyperparameters...")
    best_degree, best_alpha = tune_hyperparameters(X_trn, y_trn, X_vld, y_vld)
    
    # Train final model
    print("\n[6/7] Training final model...")
    model = train_model(X_trn, y_trn, best_degree, best_alpha)
    
    # Evaluate
    print("\n[7/7] Evaluating model...")
    train_metrics = evaluate_model(model, X_trn, y_trn, "Training")
    val_metrics = evaluate_model(model, X_vld, y_vld, "Validation")
    test_metrics = evaluate_model(model, X_tst, y_tst, "Test")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    plot_results(y_tst, test_metrics['predictions'], best_degree)
    save_metrics(train_metrics, val_metrics, test_metrics, best_degree, best_alpha)
    
    print("\n" + "="*70)
    print("POLYNOMIAL REGRESSION COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()