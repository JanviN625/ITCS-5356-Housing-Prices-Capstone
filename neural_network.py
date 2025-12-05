"""
Neural Network Model for House Price Prediction
Capstone Project - Classical ML Algorithm #3
"""

import os
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
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

def apply_feature_pipeline(X_trn: pd.DataFrame, 
                           X_vld: pd.DataFrame, 
                           X_tst: pd.DataFrame) -> Tuple:
    """Apply standardization to features"""
    scaler = Standardization()
    
    X_trn_scaled = scaler.fit_transform(X_trn)
    X_vld_scaled = scaler.transform(X_vld)
    X_tst_scaled = scaler.transform(X_tst)
    
    print(f"\nStandardization applied (μ=0, σ=1)")
    
    return X_trn_scaled, X_vld_scaled, X_tst_scaled

# ============================================================================
# Hyperparameter Tuning
# ============================================================================

def tune_hyperparameters(X_trn: pd.DataFrame, 
                         y_trn: pd.Series,
                         X_vld: pd.DataFrame,
                         y_vld: pd.Series) -> Tuple[tuple, float]:
    """Find best neural network architecture and learning rate"""
    print("\nHyperparameter tuning:")
    print("-" * 60)
    
    architectures = [
        (64,),
        (100,),
        (100, 50),
        (128, 64),
        (100, 50, 25)
    ]
    
    learning_rates = [0.001, 0.01]
    
    best_score = float('inf')
    best_params = {}
    results = []
    
    for arch in architectures:
        for lr in learning_rates:
            model = MLPRegressor(
                hidden_layer_sizes=arch,
                activation='relu',
                solver='adam',
                learning_rate_init=lr,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=False
            )
            
            model.fit(X_trn, y_trn)
            y_vld_pred = model.predict(X_vld)
            val_rmse = np.sqrt(mean_squared_error(y_vld, y_vld_pred))
            
            arch_str = ' → '.join(map(str, arch))
            results.append({
                'architecture': arch_str,
                'learning_rate': lr,
                'val_rmse': val_rmse,
                'n_iterations': model.n_iter_
            })
            
            print(f"  Architecture: {arch_str:20s} | LR: {lr:.4f} | "
                  f"Val RMSE: ${val_rmse:,.2f} | Iters: {model.n_iter_}")
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_params = {
                    'hidden_layer_sizes': arch,
                    'learning_rate': lr
                }
    
    print("-" * 60)
    arch_str = ' → '.join(map(str, best_params['hidden_layer_sizes']))
    print(f"Best architecture: {arch_str}")
    print(f"Best learning rate: {best_params['learning_rate']}")
    print(f"Best validation RMSE: ${best_score:,.2f}")
    
    # Save tuning results
    pd.DataFrame(results).to_csv('neural_network_tuning.csv', index=False)
    
    return best_params['hidden_layer_sizes'], best_params['learning_rate']

# ============================================================================
# Model Training and Evaluation
# ============================================================================

def train_model(X_trn: pd.DataFrame, 
                y_trn: pd.Series,
                hidden_layers: tuple,
                learning_rate: float) -> MLPRegressor:
    """Train Neural Network model"""
    arch_str = ' → '.join(map(str, hidden_layers))
    print(f"\nTraining Neural Network:")
    print(f"  Architecture: {X_trn.shape[1]} → {arch_str} → 1")
    print(f"  Activation: ReLU")
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: {learning_rate}")
    
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=learning_rate,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )
    
    model.fit(X_trn, y_trn)
    
    print(f"  Training completed in {model.n_iter_} iterations")
    print(f"  Early stopping: {model.t_}")
    
    return model

def evaluate_model(model: MLPRegressor, 
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
                 model: MLPRegressor,
                 save_path: str = 'neural_network_results.png'):
    """Create visualization of model performance"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price ($)', fontsize=12)
    axes[0].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[0].set_title('Neural Network: Actual vs Predicted', 
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
    
    # Loss curve
    if hasattr(model, 'loss_curve_'):
        axes[2].plot(model.loss_curve_, linewidth=2)
        axes[2].set_xlabel('Iteration', fontsize=12)
        axes[2].set_ylabel('Loss', fontsize=12)
        axes[2].set_title('Training Loss Curve', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Loss curve not available', 
                    ha='center', va='center', fontsize=12)
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {save_path}")
    plt.show()

def save_metrics(train_metrics: dict, 
                 val_metrics: dict, 
                 test_metrics: dict,
                 hidden_layers: tuple,
                 learning_rate: float,
                 n_iterations: int,
                 filepath: str = 'neural_network_metrics.csv'):
    """Save metrics to CSV file"""
    metrics_df = pd.DataFrame({
        'hidden_layers': [str(hidden_layers)],
        'learning_rate': [learning_rate],
        'n_iterations': [n_iterations],
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
    print("NEURAL NETWORK - HOUSE PRICE PREDICTION")
    print("="*70)
    
    # Load and preprocess
    print("\n[1/7] Loading and preprocessing data...")
    df = load_data('train.csv')
    df_clean = preprocess_data(df)
    
    # Split features and labels
    print("\n[2/7] Splitting features and labels...")
    X, y = feature_label_split(df_clean)
    
    # Split data
    print("\n[3/7] Splitting into train/validation/test sets...")
    X_trn, y_trn, X_vld, y_vld, X_tst, y_tst = train_valid_test_split(X, y)
    
    # Apply scaling
    print("\n[4/7] Applying standardization...")
    X_trn_scaled, X_vld_scaled, X_tst_scaled = apply_feature_pipeline(
        X_trn, X_vld, X_tst
    )
    
    # Hyperparameter tuning
    print("\n[5/7] Tuning hyperparameters...")
    best_arch, best_lr = tune_hyperparameters(
        X_trn_scaled, y_trn, X_vld_scaled, y_vld
    )
    
    # Train final model
    print("\n[6/7] Training final model...")
    model = train_model(X_trn_scaled, y_trn, best_arch, best_lr)
    
    # Evaluate
    print("\n[7/7] Evaluating model...")
    train_metrics = evaluate_model(model, X_trn_scaled, y_trn, "Training")
    val_metrics = evaluate_model(model, X_vld_scaled, y_vld, "Validation")
    test_metrics = evaluate_model(model, X_tst_scaled, y_tst, "Test")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    plot_results(y_tst, test_metrics['predictions'], model)
    save_metrics(train_metrics, val_metrics, test_metrics, 
                 best_arch, best_lr, model.n_iter_)
    
    print("\n" + "="*70)
    print("NEURAL NETWORK COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()