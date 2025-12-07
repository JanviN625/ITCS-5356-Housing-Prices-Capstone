import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

np.random.seed(42)

# -----------------------------
# Data Loading
# -----------------------------

def load_data(filepath='data/train.csv'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)


def preprocess_data(df):
    df = df.copy()
    df = df.drop('Id', axis=1, errors='ignore')
    
    y = df['SalePrice'].copy()
    X = df.drop('SalePrice', axis=1)
    
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna('None', inplace=True)
    
    X = pd.get_dummies(X, drop_first=True)
    return X, y

# -----------------------------
# Stacking Ensemble
# -----------------------------

def train_stacking_ensemble(X_train, y_train, X_test, y_test):
    print("\n" + "="*60)
    print("TRAINING STACKING ENSEMBLE (PAPER METHOD)")
    print("="*60)
    
    print("\nArchitecture:")
    print("  Level 0 (Base learners):")
    print("    • Random Forest (200 estimators, max_depth=15)")
    print("    • XGBoost (200 estimators, max_depth=5)")
    print("    • LightGBM (200 estimators, max_depth=5)")
    print("  Level 1 (Meta-learner):")
    print("    • Ridge Regression (α=1.0)")
    print("  Cross-validation: 5-fold")
    
    base_estimators = [
        ('rf', RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )),
        ('xgb', XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )),
        ('lgbm', LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ))
    ]
    
    meta_model = Ridge(alpha=1.0)
    
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    
    print("\nTraining stacked ensemble...")
    print("  Step 1: Training base learners with 5-fold CV...")
    print("  Step 2: Generating meta-features...")
    print("  Step 3: Training meta-learner...")
    stacking_model.fit(X_train, y_train)
    print("✓ Training complete")
    
    # Predictions
    y_train_pred = stacking_model.predict(X_train)
    y_test_pred = stacking_model.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nResults:")
    print(f"  Train - MSE: ${train_mse:,.0f}, RMSE: ${train_rmse:,.2f}, MAE: ${train_mae:,.2f}, R²: {train_r2:.4f}")
    print(f"  Test  - MSE: ${test_mse:,.0f}, RMSE: ${test_rmse:,.2f}, MAE: ${test_mae:,.2f}, R²: {test_r2:.4f}")
    
    return stacking_model, y_test_pred, {
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }

# -----------------------------
# Evaluation
# -----------------------------

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae}

# -----------------------------
# Visualization
# -----------------------------

def plot_results(y_true, y_pred):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Stacking Ensemble: Actual vs Predicted')
    plt.grid(True)
    
    residuals = y_true - y_pred
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('results/stacking_paper', exist_ok=True)
    plt.savefig('results/stacking_paper/stacking_paper_results.png', dpi=300)
    plt.close()

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("\n" + "="*60)
    print("STACKING ENSEMBLE (PAPER IMPLEMENTATION)")
    print("Paper: 'Housing Price Prediction via Improved ML'")
    print("Truong et al. (2020)")
    print("="*60)
    
    # Load data
    df = load_data('data/train.csv')
    print(f"\nDataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Preprocess
    X, y = preprocess_data(df)
    print(f"Features after preprocessing: {X.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train/Test split: {len(X_train)}/{len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train stacking ensemble
    stacking_model, y_pred, stacking_metrics = train_stacking_ensemble(
        X_train_scaled, y_train.values, X_test_scaled, y_test.values
    )
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_results(y_test.values, y_pred)
    print("✓ Plots saved to results/stacking_paper/")
    
    # Save metrics
    results_df = pd.DataFrame({
        'Metric': ['Method', 'Base_Models', 'Meta_Model', 
                   'Train_MSE', 'Train_RMSE', 'Train_MAE',
                   'Test_MSE', 'Test_RMSE', 'Test_MAE',
                   'Train_R2', 'Test_R2'],
        'Value': [
            'Stacking', 'RF+XGB+LGBM', 'Ridge',
            stacking_metrics['train_mse'], stacking_metrics['train_rmse'], stacking_metrics['train_mae'],
            stacking_metrics['test_mse'], stacking_metrics['test_rmse'], stacking_metrics['test_mae'],
            stacking_metrics['train_r2'], stacking_metrics['test_r2']
        ]
    })
    os.makedirs('results/stacking_paper', exist_ok=True)
    results_df.to_csv('results/stacking_paper/stacking_paper_metrics.csv', index=False)
    print("✓ Metrics saved to results/stacking_paper/stacking_paper_metrics.csv")
    
    # Summary
    print("\n" + "="*60)
    print("KEY FINDINGS (per paper):")
    print(f"  • Stacking Ensemble Test RMSE: ${stacking_metrics['test_rmse']:,.2f}")
    print(f"  • Stacking Ensemble Test R²: {stacking_metrics['test_r2']:.4f}")
    print(f"  • Method: 3 base models (RF, XGB, LGBM) + Ridge meta-learner")
    print("="*60)
    print("COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()