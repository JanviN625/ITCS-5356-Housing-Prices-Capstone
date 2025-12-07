"""
Polynomial Regression - House Price Prediction
Capstone Project - Classical ML Algorithm #2

Manual polynomial feature expansion with Ridge regularization.
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
from sklearn.linear_model import Ridge

# Set random seed
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
    
    # Handle missing values - numerical
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna(X[col].median(), inplace=True)
    
    # Handle missing values - categorical
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if X[col].isnull().sum() > 0:
            X[col].fillna('None', inplace=True)
    
    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"After preprocessing: {X.shape[1]} features")
    
    return X, y

# -----------------------------
# Feature Engineering
# -----------------------------

def select_top_features(X_train, y_train, X_test, k=15):
    """
    Select top k features based on correlation with target.
    This reduces dimensionality before polynomial expansion.
    """
    # Calculate correlation with target
    correlations = {}
    for col in X_train.columns:
        corr = np.corrcoef(X_train[col], y_train)[0, 1]
        correlations[col] = abs(corr)
    
    # Sort by correlation
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:k]
    selected_cols = [feat for feat, _ in top_features]
    
    print(f"\nTop {k} features selected:")
    for i, (feat, corr) in enumerate(top_features, 1):
        print(f"  {i}. {feat[:40]}: {corr:.4f}")
    
    return X_train[selected_cols], X_test[selected_cols], selected_cols


def build_polynomial_features(X, degree):
    """
    Manually build polynomial features up to specified degree.
    For each feature x, create: x, x^2, x^3, ..., x^degree
    """
    X_poly = np.ones((X.shape[0], 1))  # bias term
    
    # Add original features
    X_poly = np.hstack([X_poly, X])
    
    # Add polynomial terms
    if degree > 1:
        for d in range(2, degree + 1):
            X_poly = np.hstack([X_poly, X ** d])
    
    return X_poly

# -----------------------------
# Model Training
# -----------------------------

def fit_polynomial_regression(X, y, degree, alpha=1.0):
    """
    Fit polynomial regression with Ridge regularization.
    
    Args:
        X: features
        y: target
        degree: polynomial degree
        alpha: regularization strength
    """
    # Build polynomial features
    X_poly = build_polynomial_features(X, degree)
    
    print(f"Polynomial features created: {X_poly.shape[1]}")
    
    # Fit Ridge regression
    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)
    
    return model


def predict_polynomial(model, X, degree):
    """Make predictions with polynomial features"""
    X_poly = build_polynomial_features(X, degree)
    return model.predict(X_poly)


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Find best polynomial degree and regularization parameter.
    """
    print("\nHyperparameter tuning:")
    print("-" * 60)
    
    degrees = [1, 2, 3]
    alphas = [0.1, 1.0, 10.0, 100.0]
    
    best_score = float('inf')
    best_params = {}
    results = []
    
    for degree in degrees:
        for alpha in alphas:
            # Train model
            model = fit_polynomial_regression(X_train, y_train, degree, alpha)
            
            # Evaluate on validation set
            y_val_pred = predict_polynomial(model, X_val, degree)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            
            results.append({
                'degree': degree,
                'alpha': alpha,
                'val_rmse': val_rmse
            })
            
            print(f"Degree={degree}, Alpha={alpha:6.1f} -> Val RMSE: ${val_rmse:,.2f}")
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_params = {'degree': degree, 'alpha': alpha}
    
    print("-" * 60)
    print(f"Best: Degree={best_params['degree']}, Alpha={best_params['alpha']}")
    print(f"Best Val RMSE: ${best_score:,.2f}")
    
    # Save tuning results
    pd.DataFrame(results).to_csv('polynomial_regression_tuning.csv', index=False)
    
    return best_params['degree'], best_params['alpha']

# -----------------------------
# Evaluation
# -----------------------------

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

def plot_results(y_true, y_pred, degree):
    """Plot actual vs predicted and residuals"""
    plt.figure(figsize=(12, 5))
    
    # Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.grid(True)
    
    # Residuals
    residuals = y_true - y_pred
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('polynomial_regression_results.png', dpi=300)
    print("Plot saved: polynomial_regression_results.png")
    plt.show()

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("="*60)
    print("POLYNOMIAL REGRESSION - HOUSE PRICE PREDICTION")
    print("="*60)
    
    # Load data
    print("\n[Step 1] Loading data...")
    df = load_data('train.csv')
    
    # Preprocess
    print("\n[Step 2] Preprocessing...")
    X, y = preprocess_data(df)
    
    # Split data
    print("\n[Step 3] Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Feature selection
    print("\n[Step 4] Selecting top features...")
    X_train_sel, X_val_sel, selected_features = select_top_features(
        X_train, y_train, X_val, k=15
    )
    X_test_sel = X_test[selected_features]
    
    # Scale features
    print("\n[Step 5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    # Hyperparameter tuning
    print("\n[Step 6] Tuning hyperparameters...")
    best_degree, best_alpha = tune_hyperparameters(
        X_train_scaled, y_train.values,
        X_val_scaled, y_val.values
    )
    
    # Train final model
    print(f"\n[Step 7] Training final model (degree={best_degree}, alpha={best_alpha})...")
    model = fit_polynomial_regression(
        X_train_scaled, y_train.values, best_degree, best_alpha
    )
    
    # Make predictions
    print("\n[Step 8] Making predictions...")
    y_train_pred = predict_polynomial(model, X_train_scaled, best_degree)
    y_test_pred = predict_polynomial(model, X_test_scaled, best_degree)
    
    # Evaluate
    print("\n[Step 9] Evaluating model...")
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
    print("\n[Step 10] Creating visualizations...")
    plot_results(y_test.values, y_test_pred, best_degree)
    
    # Save metrics
    print("\n[Step 11] Saving results...")
    results_df = pd.DataFrame({
        'Metric': ['Degree', 'Alpha', 'Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2'],
        'Value': [
            best_degree,
            best_alpha,
            train_metrics['rmse'],
            test_metrics['rmse'],
            train_metrics['r2'],
            test_metrics['r2']
        ]
    })
    results_df.to_csv('polynomial_regression_metrics.csv', index=False)
    print("Metrics saved: polynomial_regression_metrics.csv")
    
    print("\n" + "="*60)
    print("POLYNOMIAL REGRESSION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()