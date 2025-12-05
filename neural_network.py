"""
Multi-Layer Neural Network for House Price Prediction
Capstone Project - Classical ML Algorithm #3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
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

def train_neural_network(X_train, y_train, X_test, y_test, 
                         hidden_layers=(100, 50), activation='relu', 
                         learning_rate=0.001, max_iter=500):
    """
    Train Multi-Layer Neural Network
    """
    print("\n" + "="*60)
    print(f"TRAINING NEURAL NETWORK")
    print(f"Architecture: {X_train.shape[1]} -> {' -> '.join(map(str, hidden_layers))} -> 1")
    print(f"Activation: {activation}, Learning Rate: {learning_rate}")
    print("="*60)
    
    # Standardize features (critical for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver='adam',
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )
    
    print("\nTraining neural network...")
    model.fit(X_train_scaled, y_train)
    
    print(f"Training completed in {model.n_iter_} iterations")
    print(f"Best validation score: {model.best_validation_score_:.4f}")
    
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
    
    print(f"\nTraining Metrics:")
    print(f"  RMSE: ${train_rmse:,.2f}")
    print(f"  MAE:  ${train_mae:,.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  RMSE: ${test_rmse:,.2f}")
    print(f"  MAE:  ${test_mae:,.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    return model, scaler, y_test_pred, {
        'hidden_layers': hidden_layers,
        'activation': activation,
        'learning_rate': learning_rate,
        'n_iterations': model.n_iter_,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }

def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """
    Test different neural network architectures and hyperparameters
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
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
    
    print("\nTesting different architectures and learning rates...")
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
            
            model.fit(X_train_scaled, y_train)
            y_val_pred = model.predict(X_val_scaled)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            
            arch_str = ' -> '.join(map(str, arch))
            results.append({
                'architecture': arch_str,
                'learning_rate': lr,
                'val_rmse': val_rmse,
                'n_iterations': model.n_iter_
            })
            
            print(f"  Architecture: {arch_str:20} | LR: {lr:6.4f} | Val RMSE: ${val_rmse:,.2f} | Iters: {model.n_iter_}")
            
            if val_rmse < best_score:
                best_score = val_rmse
                best_params = {
                    'hidden_layers': arch,
                    'learning_rate': lr
                }
    
    print(f"\n✓ Best architecture: {' -> '.join(map(str, best_params['hidden_layers']))}")
    print(f"  Best learning rate: {best_params['learning_rate']}")
    print(f"  Best validation RMSE: ${best_score:,.2f}")
    
    return best_params, pd.DataFrame(results)

def plot_results(y_test, y_pred, metrics, loss_curve=None):
    """
    Visualize model performance
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Price ($)', fontsize=12)
    axes[0].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[0].set_title('Neural Network: Actual vs Predicted', fontsize=14, fontweight='bold')
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
    
    # Loss curve
    if loss_curve is not None:
        axes[2].plot(loss_curve, linewidth=2)
        axes[2].set_xlabel('Iteration', fontsize=12)
        axes[2].set_ylabel('Loss', fontsize=12)
        axes[2].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Loss curve not available', 
                    ha='center', va='center', fontsize=12)
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('neural_network_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plots saved as 'neural_network_results.png'")
    plt.show()

def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("NEURAL NETWORK - HOUSE PRICE PREDICTION")
    print("="*60)
    
    # Load data
    X, y = load_and_preprocess_data('train.csv')
    
    # Split data (train/val/test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42
    )
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Hyperparameter tuning
    best_params, tuning_results = hyperparameter_tuning(X_train, y_train, X_val, y_val)
    tuning_results.to_csv('neural_network_tuning.csv', index=False)
    
    # Combine train and validation for final training
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # Train final model with best parameters
    model, scaler, y_pred, metrics = train_neural_network(
        X_train_full, y_train_full, X_test, y_test,
        hidden_layers=best_params['hidden_layers'],
        learning_rate=best_params['learning_rate']
    )
    
    # Plot results (with loss curve if available)
    loss_curve = model.loss_curve_ if hasattr(model, 'loss_curve_') else None
    plot_results(y_test, y_pred, metrics, loss_curve)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('neural_network_metrics.csv', index=False)
    print("\n💾 Metrics saved to 'neural_network_metrics.csv'")
    print("💾 Tuning results saved to 'neural_network_tuning.csv'")
    
    print("\n" + "="*60)
    print("NEURAL NETWORK COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()