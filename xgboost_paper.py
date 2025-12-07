import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

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
# Hyperparameter Tuning
# -----------------------------

def tune_hyperparameters(X_train, y_train):
    print("\nHyperparameter tuning (GridSearchCV)...")
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 3, 5]
    }
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"  Testing {total_combinations} combinations (5-fold CV)")
    print("  Search space:")
    print("    • max_depth: [3, 5, 7]")
    print("    • learning_rate: [0.01, 0.05, 0.1]")
    print("    • n_estimators: [100, 500, 1000]")
    print("    • min_child_weight: [1, 3, 5]")
    print("    • subsample: [0.8]")
    print("    • colsample_bytree: [0.8]")
    
    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=0,  # Silent mode
        n_jobs=-1
    )
    
    print("\n  Training in progress...")
    grid_search.fit(X_train, y_train)
    
    print("\n  Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    
    best_score = np.sqrt(-grid_search.best_score_)
    print(f"    Best CV RMSE: ${best_score:,.2f}")
    print("✓ Tuning complete")
    
    # Save top 10 results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_rmse'] = np.sqrt(-results_df['mean_test_score'])
    results_df = results_df.sort_values('mean_rmse')
    top_10 = results_df[['params', 'mean_rmse']].head(10)
    os.makedirs('results/xgboost_paper', exist_ok=True)
    top_10.to_csv('results/xgboost_paper/xgboost_paper_tuning.csv', index=False)
    
    return grid_search.best_params_

# -----------------------------
# Model Training
# -----------------------------

def train_xgboost(X_train, y_train, best_params):
    print(f"\nTraining XGBoost with optimal parameters...")
    
    model = XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✓ Training complete")
    return model


def analyze_feature_importance(model, feature_names, top_n=10):
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} most important features:")
    for i, (feat, imp) in enumerate(feature_importance_df.head(top_n).values, 1):
        feat_short = feat[:35] + "..." if len(feat) > 35 else feat
        print(f"  {i:2d}. {feat_short:40s} {imp:.4f}")
    print(f"(Full list of {len(feature_names)} features saved to CSV)")
    
    os.makedirs('results/xgboost_paper', exist_ok=True)
    feature_importance_df.to_csv('results/xgboost_paper/xgboost_paper_feature_importance.csv', index=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('XGBoost: Top 15 Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('results/xgboost_paper/xgboost_paper_feature_importance.png', dpi=300)
    plt.close()
    
    return feature_importance_df

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
    plt.title('XGBoost (Paper): Actual vs Predicted')
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
    os.makedirs('results/xgboost_paper', exist_ok=True)
    plt.savefig('results/xgboost_paper/xgboost_paper_results.png', dpi=300)
    plt.close()

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("\n" + "="*60)
    print("XGBOOST (PAPER IMPLEMENTATION)")
    print("Paper: 'An Optimal House Price Prediction Algorithm'")
    print("Phan, T.D. (2024)")
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
    
    # Tune hyperparameters
    best_params = tune_hyperparameters(X_train_scaled, y_train.values)
    
    # Train model
    model = train_xgboost(X_train_scaled, y_train.values, best_params)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate
    train_metrics = evaluate_model(y_train.values, y_train_pred)
    test_metrics = evaluate_model(y_test.values, y_test_pred)
    
    print("\nResults:")
    print(f"  Train - MSE: ${train_metrics['mse']:,.0f}, RMSE: ${train_metrics['rmse']:,.2f}, MAE: ${train_metrics['mae']:,.2f}, R²: {train_metrics['r2']:.4f}")
    print(f"  Test  - MSE: ${test_metrics['mse']:,.0f}, RMSE: ${test_metrics['rmse']:,.2f}, MAE: ${test_metrics['mae']:,.2f}, R²: {test_metrics['r2']:.4f}")
    
    # Feature importance
    feature_importance_df = analyze_feature_importance(model, X.columns, top_n=10)
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_results(y_test.values, y_test_pred)
    print("✓ Plots saved to results/xgboost_paper/")
    
    # Save metrics
    results_df = pd.DataFrame({
        'Metric': ['Method', 'Max_Depth', 'Learning_Rate', 'N_Estimators', 
                   'Train_MSE', 'Train_RMSE', 'Train_MAE',
                   'Test_MSE', 'Test_RMSE', 'Test_MAE',
                   'Train_R2', 'Test_R2'],
        'Value': [
            'XGBoost', best_params['max_depth'], best_params['learning_rate'], best_params['n_estimators'],
            train_metrics['mse'], train_metrics['rmse'], train_metrics['mae'],
            test_metrics['mse'], test_metrics['rmse'], test_metrics['mae'],
            train_metrics['r2'], test_metrics['r2']
        ]
    })
    os.makedirs('results/xgboost_paper', exist_ok=True)
    results_df.to_csv('results/xgboost_paper/xgboost_paper_metrics.csv', index=False)
    print("✓ Metrics saved to results/xgboost_paper/xgboost_paper_metrics.csv")
    
    # Summary
    print("\n" + "="*60)
    print("KEY FINDINGS (per paper):")
    print(f"  • Best Test RMSE: ${test_metrics['rmse']:,.2f}")
    print(f"  • Best Test R²: {test_metrics['r2']:.4f}")
    top_3 = ', '.join(feature_importance_df.head(3)['feature'].tolist())
    print(f"  • Top 3 features: {top_3}")
    print("="*60)
    print("COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()