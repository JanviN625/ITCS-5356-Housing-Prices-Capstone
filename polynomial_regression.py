"""
Polynomial Regression Model for House Price Prediction
Capstone Project - Classical ML Algorithm #2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
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
    
    # Separate features and target
    target = 'SalePrice'
    y = df[target]
    X = df.drop([target, 'Id'], axis=1, errors='ignore')
    
    # Handle missing values intelligently
    cat_features = X.select_dtypes(include=['object']).columns
    for col in cat_features:
        X[col] = X[col].fillna('None')
    
    num_features = X.select_dtypes(include=[np.number]).columns
    X[num_features] = X[num_features].fillna(X[num_features].median())
    
    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"After preprocessing: {X.shape}")
    print(f"Target (SalePrice) range: ${y.min():,.0f} - ${y.max():,.0f}")
    
    return X, y

def select_important_features(X, y, n_features=10):
    """
    Select most important features to reduce dimensionality before polynomial expansion
    This prevents exponential growth of features
    """
    from sklearn.ensemble import RandomForestRegressor
    
    print(f"\nSelecting top {n_features} features to prevent dimensionality explosion...")
    
    # Use Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importances.head(n_features)['feature'].tolist()
    
    print(f"\nTop {n_features} features selected:")
    for i, (feat, imp) in enumerate(zip(top_features, importances.head(n_features)['importance']), 1):
        print(f"  {i}. {feat}: {imp:.4f}")
    
    return X[top_features]

def train_polynomial_regression(X_train, y_train, X_test, y_test, degree=2, alpha=1.0):
    """
    Train Polynomial Regression model with Ridge regularization
    """
    print("\n" + "="*60)
    print(f"TRAINING POLYNOMIAL REGRESSION (Degree={degree}, Alpha={alpha})")
    print("="*60)
    
    # Create pipeline: Polynomial Features -> Scaling -> Ridge Regression
    # Using Ridge instead of LinearRegression to handle multicollinearity
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=alpha))
    ])
    
    # Train model
    print("\nTraining model (this may take a moment)...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
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
    print("Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, 
                                 cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
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
    
    # Get number of polynomial features created
    n_poly_features = model.named_steps['poly'].n_output_features_
    print(f"\nNumber of polynomial features created: {n_poly_features}")
    
    return model, y_test_pred, {
        'degree': degree,
        'alpha': alpha,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_rmse': cv_rmse,
        'n_poly_features': n_poly_features
    }

def hyperparameter_tuning(X_train, y_train):
    """
    Find best polynomial degree and regularization parameter
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    degrees = [1, 2, 3]
    alphas = [0.1, 1.0, 10.0, 100.0]
    
    best_score = float('inf')
    best_params = {}
    
    results = []
    
    print("\nTesting different combinations of degree and alpha...")
    for degree in degrees:
        for alpha in alphas:
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('scaler', StandardScaler()),
                ('regressor', Ridge(alpha=alpha))
            ])
            
            cv_scores = cross_val_score(model, X_train, y_train, 
                                        cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results.append({
                'degree': degree,
                'alpha': alpha,
                'cv_rmse': cv_rmse
            })
            
            print(f"  Degree={degree}, Alpha={alpha:6.1f} -> CV RMSE: ${cv_rmse:,.2f}")
            
            if cv_rmse < best_score:
                best_score = cv_rmse
                best_params = {'degree': degree, 'alpha': alpha}
    
    print(f"\n✓ Best parameters: Degree={best_params['degree']}, Alpha={best_params['alpha']}")
    print(f"  Best CV RMSE: ${best_score:,.2f}")
    
    return best_params, pd.DataFrame(results)

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
    axes[0].set_title(f'Polynomial Regression (Degree={metrics["degree"]}): Actual vs Predicted', 
                     fontsize=14, fontweight='bold')
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
    plt.savefig('polynomial_regression_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plots saved as 'polynomial_regression_results.png'")
    plt.show()

def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("POLYNOMIAL REGRESSION - HOUSE PRICE PREDICTION")
    print("="*60)
    
    # Load data
    X, y = load_and_preprocess_data('train.csv')
    
    # Select important features to prevent dimensionality explosion
    X_selected = select_important_features(X, y, n_features=10)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Hyperparameter tuning
    best_params, tuning_results = hyperparameter_tuning(X_train, y_train)
    tuning_results.to_csv('polynomial_regression_tuning.csv', index=False)
    
    # Train final model with best parameters
    model, y_pred, metrics = train_polynomial_regression(
        X_train, y_train, X_test, y_test,
        degree=best_params['degree'],
        alpha=best_params['alpha']
    )
    
    # Plot results
    plot_results(y_test, y_pred, metrics)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('polynomial_regression_metrics.csv', index=False)
    print("\n💾 Metrics saved to 'polynomial_regression_metrics.csv'")
    print("💾 Tuning results saved to 'polynomial_regression_tuning.csv'")
    
    print("\n" + "="*60)
    print("POLYNOMIAL REGRESSION COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()