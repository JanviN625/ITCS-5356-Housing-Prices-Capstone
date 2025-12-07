"""
Linear Regression - House Price Prediction
Capstone Project - Classical ML Algorithm #1

Implementation using both closed-form solution and gradient descent.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# -----------------------------
# Data Loading
# -----------------------------

def load_data(filepath='train.csv'):
    """Load the Ames Housing dataset"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    """Clean and prepare data"""
    df = df.copy()
    
    # Drop ID column
    df = df.drop('Id', axis=1, errors='ignore')
    
    # Separate target
    y = df['SalePrice'].copy()
    X = df.drop('SalePrice', axis=1)
    
    # Handle missing values - numerical features
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Handle missing values - categorical features
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna('None', inplace=True)
    
    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"After preprocessing: {X.shape[1]} features")
    
    return X, y

# -----------------------------
# Model Implementation
# -----------------------------

def fit_linear_regression_closed_form(X, y):
    """
    Fit linear regression using closed-form solution.
    theta = (X^T X)^(-1) X^T y
    """
    # Add bias term
    X_bias = np.c_[np.ones(X.shape[0]), X]
    
    # Closed-form solution
    theta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
    
    return theta


def fit_linear_regression_gd(X, y, lr=0.01, epochs=1000):
    """
    Fit linear regression using gradient descent.
    """
    # Add bias term
    X_bias = np.c_[np.ones(X.shape[0]), X]
    
    # Initialize parameters
    theta = np.zeros(X_bias.shape[1])
    m = len(y)
    
    losses = []
    
    for epoch in range(epochs):
        # Predictions
        preds = X_bias @ theta
        
        # Compute loss
        loss = np.mean((preds - y) ** 2)
        losses.append(loss)
        
        # Gradient
        gradient = (1/m) * X_bias.T @ (preds - y)
        
        # Update parameters
        theta -= lr * gradient
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.2f}")
    
    return theta, losses


def predict_linear(X, theta):
    """Make predictions using linear regression"""
    # Add bias term
    X_bias = np.c_[np.ones(X.shape[0]), X]
    return X_bias @ theta


def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mae': mae
    }

# -----------------------------
# Visualization
# -----------------------------

def plot_predictions(y_true, y_pred, title="Linear Regression Results"):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted')
    plt.grid(True)
    
    # Subplot 2: Residuals
    residuals = y_true - y_pred
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('linear_regression_results.png', dpi=300)
    print("Plot saved: linear_regression_results.png")
    plt.show()


def plot_training_curve(losses):
    """Plot training loss curve"""
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('linear_regression_training.png', dpi=300)
    print("Plot saved: linear_regression_training.png")
    plt.show()

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("="*60)
    print("LINEAR REGRESSION - HOUSE PRICE PREDICTION")
    print("="*60)
    
    # Load data
    print("\n[Step 1] Loading data...")
    df = load_data('train.csv')
    
    # Preprocess
    print("\n[Step 2] Preprocessing...")
    X, y = preprocess_data(df)
    
    # Split data
    print("\n[Step 3] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale features
    print("\n[Step 4] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model using closed-form solution
    print("\n[Step 5] Training model (Closed-Form Solution)...")
    theta = fit_linear_regression_closed_form(X_train_scaled, y_train.values)
    print(f"Parameters learned: {len(theta)}")
    
    # Make predictions
    print("\n[Step 6] Making predictions...")
    y_train_pred = predict_linear(X_train_scaled, theta)
    y_test_pred = predict_linear(X_test_scaled, theta)
    
    # Evaluate
    print("\n[Step 7] Evaluating model...")
    train_metrics = evaluate_model(y_train.values, y_train_pred)
    test_metrics = evaluate_model(y_test.values, y_test_pred)
    
    print("\nTraining Set:")
    print(f"  RMSE: ${train_metrics['rmse']:,.2f}")
    print(f"  MAE:  ${train_metrics['mae']:,.2f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")
    
    print("\nTest Set:")
    print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
    print(f"  MAE:  ${test_metrics['mae']:,.2f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    
    # Visualize
    print("\n[Step 8] Creating visualizations...")
    plot_predictions(y_test.values, y_test_pred)
    
    # Optional: Train with gradient descent to show convergence
    print("\n[Optional] Training with Gradient Descent...")
    theta_gd, losses = fit_linear_regression_gd(
        X_train_scaled, y_train.values, lr=0.1, epochs=500
    )
    plot_training_curve(losses)
    
    # Save metrics
    print("\n[Step 9] Saving results...")
    results_df = pd.DataFrame({
        'Method': ['Closed-Form', 'Gradient Descent'],
        'Train_RMSE': [train_metrics['rmse'], 
                       np.sqrt(mean_squared_error(y_train.values, 
                                                  predict_linear(X_train_scaled, theta_gd)))],
        'Test_RMSE': [test_metrics['rmse'],
                      np.sqrt(mean_squared_error(y_test.values,
                                                 predict_linear(X_test_scaled, theta_gd)))],
        'Train_R2': [train_metrics['r2'],
                     r2_score(y_train.values, predict_linear(X_train_scaled, theta_gd))],
        'Test_R2': [test_metrics['r2'],
                    r2_score(y_test.values, predict_linear(X_test_scaled, theta_gd))]
    })
    results_df.to_csv('linear_regression_metrics.csv', index=False)
    print("Metrics saved: linear_regression_metrics.csv")
    
    print("\n" + "="*60)
    print("LINEAR REGRESSION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()