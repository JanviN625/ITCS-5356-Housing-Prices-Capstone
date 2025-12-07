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
# Feature Engineering
# -----------------------------

def select_top_features(X_train, y_train, X_test, k=15):
    correlations = {}
    for col in X_train.columns:
        corr = np.corrcoef(X_train[col], y_train)[0, 1]
        correlations[col] = abs(corr)
    
    top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:k]
    selected_cols = [feat for feat, _ in top_features]
    
    return X_train[selected_cols], X_test[selected_cols], selected_cols, top_features


def build_polynomial_features(X, degree):
    X_poly = np.ones((X.shape[0], 1))
    X_poly = np.hstack([X_poly, X])
    
    if degree > 1:
        for d in range(2, degree + 1):
            X_poly = np.hstack([X_poly, X ** d])
    
    return X_poly

# -----------------------------
# Model Training
# -----------------------------

def fit_polynomial_regression(X, y, degree, alpha=1.0):
    X_poly = build_polynomial_features(X, degree)
    model = Ridge(alpha=alpha)
    model.fit(X_poly, y)
    return model


def predict_polynomial(model, X, degree):
    X_poly = build_polynomial_features(X, degree)
    return model.predict(X_poly)


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    print("\nHyperparameter tuning...")
    print("  Testing degrees: [1, 2, 3]")
    print("  Testing alphas: [0.1, 1.0, 10.0, 100.0]")
    
    degrees = [1, 2, 3]
    alphas = [0.1, 1.0, 10.0, 100.0]
    
    best_score = float('inf')
    best_params = {}
    results = []
    
    for degree in degrees:
        for alpha in alphas:
            model = fit_polynomial_regression(X_train, y_train, degree, alpha)
            y_val_pred = predict_polynomial(model, X_val, degree)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            
            results.append({'degree': degree, 'alpha': alpha, 'val_rmse': val_rmse})
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_params = {'degree': degree, 'alpha': alpha}
    
    print("\n  Best parameters:")
    print(f"    Degree: {best_params['degree']}")
    print(f"    Alpha: {best_params['alpha']}")
    print(f"    Validation RMSE: ${best_score:,.2f}")
    print("✓ Tuning complete")
    
    os.makedirs('results/polynomial', exist_ok=True)
    pd.DataFrame(results).to_csv('results/polynomial/polynomial_regression_tuning.csv', index=False)
    
    return best_params['degree'], best_params['alpha']

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

def plot_results(y_true, y_pred, degree):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Polynomial Regression (Degree {degree})')
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
    os.makedirs('results/polynomial', exist_ok=True)
    plt.savefig('results/polynomial/polynomial_regression_results.png', dpi=300)
    plt.close()

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("\n" + "="*60)
    print("POLYNOMIAL REGRESSION - HOUSE PRICE PREDICTION")
    print("="*60)
    
    # Load data
    df = load_data('data/train.csv')
    print(f"\nDataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Preprocess
    X, y = preprocess_data(df)
    print(f"Features after preprocessing: {X.shape[1]}")
    
    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    print(f"Train/Val/Test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    
    # Feature selection
    print("\nSelecting top 15 features by correlation...")
    X_train_sel, X_val_sel, selected_features, top_features = select_top_features(
        X_train, y_train, X_val, k=15
    )
    X_test_sel = X_test[selected_features]
    
    print("\nTop 10 features:")
    for i, (feat, corr) in enumerate(top_features[:10], 1):
        feat_short = feat[:35] + "..." if len(feat) > 35 else feat
        print(f"  {i:2d}. {feat_short:40s} {corr:.4f}")
    print("(Full list of 15 features used in model)")
    print("✓ Feature selection complete")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    
    # Tune hyperparameters
    best_degree, best_alpha = tune_hyperparameters(
        X_train_scaled, y_train.values, X_val_scaled, y_val.values
    )
    
    # Train final model
    print(f"\nTraining final model (degree={best_degree}, alpha={best_alpha})...")
    model = fit_polynomial_regression(X_train_scaled, y_train.values, best_degree, best_alpha)
    X_poly = build_polynomial_features(X_train_scaled, best_degree)
    print(f"  Polynomial features created: {X_poly.shape[1]}")
    print("✓ Training complete")
    
    # Evaluate
    y_train_pred = predict_polynomial(model, X_train_scaled, best_degree)
    y_test_pred = predict_polynomial(model, X_test_scaled, best_degree)
    
    train_metrics = evaluate_model(y_train.values, y_train_pred)
    test_metrics = evaluate_model(y_test.values, y_test_pred)
    
    print("\nResults:")
    print(f"  Train - MSE: ${train_metrics['mse']:,.0f}, RMSE: ${train_metrics['rmse']:,.2f}, MAE: ${train_metrics['mae']:,.2f}, R²: {train_metrics['r2']:.4f}")
    print(f"  Test  - MSE: ${test_metrics['mse']:,.0f}, RMSE: ${test_metrics['rmse']:,.2f}, MAE: ${test_metrics['mae']:,.2f}, R²: {test_metrics['r2']:.4f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_results(y_test.values, y_test_pred, best_degree)
    print("✓ Plots saved to results/polynomial/")
    
    # Save metrics
    results_df = pd.DataFrame({
        'Metric': ['Degree', 'Alpha', 'Train_MSE', 'Train_RMSE', 'Train_MAE', 'Test_MSE', 'Test_RMSE', 'Test_MAE', 'Train_R2', 'Test_R2'],
        'Value': [
            best_degree, best_alpha, 
            train_metrics['mse'], train_metrics['rmse'], train_metrics['mae'],
            test_metrics['mse'], test_metrics['rmse'], test_metrics['mae'],
            train_metrics['r2'], test_metrics['r2']
        ]
    })
    os.makedirs('results/polynomial', exist_ok=True)
    results_df.to_csv('results/polynomial/polynomial_regression_metrics.csv', index=False)
    print("✓ Metrics saved to results/polynomial/polynomial_regression_metrics.csv")
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()