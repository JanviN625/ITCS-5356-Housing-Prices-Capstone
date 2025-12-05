"""
Linear Regression Model for House Price Prediction
Capstone Project - Classical ML Algorithm #1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(filepath='train.csv'):
    """
    Load and preprocess the Ames Housing dataset
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}, Samples: {df.shape[0]}")
    
    # Separate features and target
    target = 'SalePrice'
    y = df[target]
    X = df.drop([target, 'Id'], axis=1, errors='ignore')
    
    # Handle missing values intelligently
    # For categorical: 'NA' often means absence of feature (e.g., no pool, no garage)
    cat_features = X.select_dtypes(include=['object']).columns
    for col in cat_features:
        X[col] = X[col].fillna('None')
    
    # For numerical: use median imputation
    num_features = X.select_dtypes(include=[np.number]).columns
    X[num_features] = X[num_features].fillna(X[num_features].median())
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"\nAfter preprocessing:")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Number of features: {X.shape[1]}")
    print(f"  Target (SalePrice) range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"  Target mean: ${y.mean():,.0f}, median: ${y.median():,.0f}")
    
    return X, y

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Train Linear Regression model
    """
    print("\n" + "="*60)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                 cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    print(f"\nTraining Metrics:")
    print(f"  RMSE: ${train_rmse:,.2f}")
    print(f"  MAE:  ${train_mae:,.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  RMSE: ${test_rmse:,.2f}")
    print(f"  MAE:  ${test_mae:,.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    print(f"\n5-Fold Cross-Validation RMSE: ${cv_rmse:,.2f}")
    
    return model, scaler, y_test_pred, {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_rmse': cv_rmse
    }

def plot_results(y_test, y_pred, metrics):
    """
    Visualize model performance
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price ($)', fontsize=12)
    axes[0].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[0].set_title('Linear Regression: Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Price ($)', fontsize=12)
    axes[1].set_ylabel('Residuals ($)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plots saved as 'linear_regression_results.png'")
    plt.show()

def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("LINEAR REGRESSION - HOUSE PRICE PREDICTION")
    print("="*60)
    
    # Load data
    X, y = load_and_preprocess_data('train.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train model
    model, scaler, y_pred, metrics = train_linear_regression(
        X_train, y_train, X_test, y_test
    )
    
    # Plot results
    plot_results(y_test, y_pred, metrics)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('linear_regression_metrics.csv', index=False)
    print("\n💾 Metrics saved to 'linear_regression_metrics.csv'")
    
    print("\n" + "="*60)
    print("LINEAR REGRESSION COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()