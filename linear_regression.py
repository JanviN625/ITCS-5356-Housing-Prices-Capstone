import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
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
    
    # Remove outliers
    lower = y.quantile(0.01)
    upper = y.quantile(0.99)
    mask = (y >= lower) & (y <= upper)
    X = X[mask]
    y = y[mask]
    
    # Log transform skewed features
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].skew() > 0.75:
            X[col] = np.log1p(X[col])
    
    # Handle missing values
    for col in num_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna('None', inplace=True)
    
    # Feature engineering
    if 'OverallQual' in X.columns and 'GrLivArea' in X.columns:
        X['QualArea'] = X['OverallQual'] * X['GrLivArea']
    if 'TotalBsmtSF' in X.columns and 'GrLivArea' in X.columns:
        X['TotalSF'] = X['TotalBsmtSF'] + X['GrLivArea']
    
    X = pd.get_dummies(X, drop_first=True)
    return X, y

# -----------------------------
# Model Implementation
# -----------------------------

def fit_linear_regression_closed_form(X, y, alpha=10.0):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    I = np.eye(X_bias.shape[1])
    I[0, 0] = 0
    theta = np.linalg.pinv(X_bias.T @ X_bias + alpha * I) @ X_bias.T @ y
    return theta


def fit_linear_regression_gd(X, y, lr=0.01, epochs=1000, alpha=10.0):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros(X_bias.shape[1])
    m = len(y)
    losses = []
    
    for epoch in range(epochs):
        preds = X_bias @ theta
        mse_loss = np.mean((preds - y) ** 2)
        ridge_penalty = alpha * np.sum(theta[1:] ** 2)
        loss = mse_loss + ridge_penalty
        losses.append(loss)
        
        gradient = (1/m) * X_bias.T @ (preds - y)
        gradient[1:] += (2 * alpha / m) * theta[1:]
        theta -= lr * gradient
    
    return theta, losses


def predict_linear(X, theta):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    return X_bias @ theta


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return {'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae}

# -----------------------------
# Visualization
# -----------------------------

def plot_predictions(y_true, y_pred, title="Linear Regression Results"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted')
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
    os.makedirs('results/linear', exist_ok=True)
    plt.savefig('results/linear/linear_regression_results.png', dpi=300)
    plt.close()


def plot_training_curve(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    os.makedirs('results/linear', exist_ok=True)
    plt.savefig('results/linear/linear_regression_training.png', dpi=300)
    plt.close()

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("\n" + "="*60)
    print("LINEAR REGRESSION - HOUSE PRICE PREDICTION")
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
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train closed-form
    print("\nTraining closed-form solution (Ridge α=10)...")
    theta = fit_linear_regression_closed_form(X_train_scaled, y_train.values, alpha=10.0)
    print("✓ Training complete")
    
    # Evaluate closed-form
    y_train_pred = predict_linear(X_train_scaled, theta)
    y_test_pred = predict_linear(X_test_scaled, theta)
    
    train_metrics = evaluate_model(y_train.values, y_train_pred)
    test_metrics = evaluate_model(y_test.values, y_test_pred)
    
    print("\nClosed-Form Results:")
    print(f"  Train - MSE: ${train_metrics['mse']:,.0f}, RMSE: ${train_metrics['rmse']:,.2f}, MAE: ${train_metrics['mae']:,.2f}, R²: {train_metrics['r2']:.4f}")
    print(f"  Test  - MSE: ${test_metrics['mse']:,.0f}, RMSE: ${test_metrics['rmse']:,.2f}, MAE: ${test_metrics['mae']:,.2f}, R²: {test_metrics['r2']:.4f}")
    print(f"  Overfitting gap: {train_metrics['r2'] - test_metrics['r2']:.4f}")
    
    # Train gradient descent
    print("\nTraining gradient descent (Ridge α=10)...")
    theta_gd, losses = fit_linear_regression_gd(X_train_scaled, y_train.values, lr=0.1, epochs=500, alpha=10.0)
    print("✓ Training complete")
    
    # Evaluate gradient descent
    y_train_pred_gd = predict_linear(X_train_scaled, theta_gd)
    y_test_pred_gd = predict_linear(X_test_scaled, theta_gd)
    
    train_metrics_gd = evaluate_model(y_train.values, y_train_pred_gd)
    test_metrics_gd = evaluate_model(y_test.values, y_test_pred_gd)
    
    print("\nGradient Descent Results:")
    print(f"  Train - MSE: ${train_metrics_gd['mse']:,.0f}, RMSE: ${train_metrics_gd['rmse']:,.2f}, MAE: ${train_metrics_gd['mae']:,.2f}, R²: {train_metrics_gd['r2']:.4f}")
    print(f"  Test  - MSE: ${test_metrics_gd['mse']:,.0f}, RMSE: ${test_metrics_gd['rmse']:,.2f}, MAE: ${test_metrics_gd['mae']:,.2f}, R²: {test_metrics_gd['r2']:.4f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_predictions(y_test.values, y_test_pred)
    plot_training_curve(losses)
    print("✓ Plots saved to results/linear/")
    
    # Save metrics
    results_df = pd.DataFrame({
        'Method': ['Closed-Form', 'Gradient Descent'],
        'Alpha': [10.0, 10.0],
        'Train_MSE': [train_metrics['mse'], train_metrics_gd['mse']],
        'Train_RMSE': [train_metrics['rmse'], train_metrics_gd['rmse']],
        'Train_MAE': [train_metrics['mae'], train_metrics_gd['mae']],
        'Test_MSE': [test_metrics['mse'], test_metrics_gd['mse']],
        'Test_RMSE': [test_metrics['rmse'], test_metrics_gd['rmse']],
        'Test_MAE': [test_metrics['mae'], test_metrics_gd['mae']],
        'Train_R2': [train_metrics['r2'], train_metrics_gd['r2']],
        'Test_R2': [test_metrics['r2'], test_metrics_gd['r2']]
    })
    os.makedirs('results/linear', exist_ok=True)
    results_df.to_csv('results/linear/linear_regression_metrics.csv', index=False)
    print("✓ Metrics saved to results/linear/linear_regression_metrics.csv")
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()