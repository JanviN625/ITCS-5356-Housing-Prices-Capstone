import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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
# Activation Functions
# -----------------------------

def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def identity(Z):
    return Z


def identity_derivative(Z):
    return np.ones_like(Z)

# -----------------------------
# Neural Network Implementation
# -----------------------------

def initialize_parameters(layer_dims):
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l-1], layer_dims[l]) * 0.01
        parameters[f'b{l}'] = np.zeros((1, layer_dims[l]))
    
    return parameters


def forward_propagation(X, parameters):
    cache = {}
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        Z = A @ parameters[f'W{l}'] + parameters[f'b{l}']
        A = relu(Z)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
    
    Z_out = A @ parameters[f'W{L}'] + parameters[f'b{L}']
    A_out = identity(Z_out)
    cache[f'Z{L}'] = Z_out
    cache[f'A{L}'] = A_out
    
    return cache, A_out


def backward_propagation(X, y, parameters, cache):
    grads = {}
    m = X.shape[0]
    L = len(parameters) // 2
    
    A_out = cache[f'A{L}']
    dZ = A_out - y.reshape(-1, 1)
    
    for l in range(L, 0, -1):
        if l == 1:
            A_prev = X
        else:
            A_prev = cache[f'A{l-1}']
        
        grads[f'dW{l}'] = (1/m) * A_prev.T @ dZ
        grads[f'db{l}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        
        if l > 1:
            dA_prev = dZ @ parameters[f'W{l}'].T
            dZ = dA_prev * relu_derivative(cache[f'Z{l-1}'])
    
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    
    return parameters


def compute_cost(y_true, y_pred):
    m = len(y_true)
    cost = (1/(2*m)) * np.sum((y_pred.flatten() - y_true) ** 2)
    return cost

# -----------------------------
# Model Training
# -----------------------------

def train_neural_network(X, y, layer_dims, learning_rate=0.01, epochs=1000, verbose=False):
    parameters = initialize_parameters(layer_dims)
    costs = []
    
    for epoch in range(epochs):
        cache, A_out = forward_propagation(X, parameters)
        cost = compute_cost(y, A_out)
        costs.append(cost)
        grads = backward_propagation(X, y, parameters, cache)
        parameters = update_parameters(parameters, grads, learning_rate)
    
    return parameters, costs


def predict_neural_network(X, parameters):
    cache, A_out = forward_propagation(X, parameters)
    return A_out.flatten()

# -----------------------------
# Hyperparameter Tuning
# -----------------------------

def tune_architecture(X_train, y_train, X_val, y_val, learning_rate=0.01):
    print("\nTuning network architecture...")
    
    input_dim = X_train.shape[1]
    architectures = [
        [input_dim, 64, 1],
        [input_dim, 100, 1],
        [input_dim, 128, 1],
        [input_dim, 100, 50, 1],
        [input_dim, 128, 64, 1]
    ]
    
    print(f"  Testing architectures: [64], [100], [128], [100,50], [128,64]")
    
    best_score = float('inf')
    best_arch = None
    results = []
    
    for arch in architectures:
        params, costs = train_neural_network(X_train, y_train, arch, learning_rate, epochs=500, verbose=False)
        y_val_pred = predict_neural_network(X_val, params)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        arch_str = ' -> '.join(map(str, arch))
        results.append({'architecture': arch_str, 'val_rmse': val_rmse})
        
        if val_rmse < best_score:
            best_score = val_rmse
            best_arch = arch
    
    best_arch_str = ' -> '.join(map(str, best_arch))
    print(f"\n  Best architecture: {best_arch_str}")
    print(f"  Best validation RMSE: ${best_score:,.2f}")
    print("✓ Tuning complete")
    
    os.makedirs('results/neural_network', exist_ok=True)
    pd.DataFrame(results).to_csv('results/neural_network/neural_network_tuning.csv', index=False)
    
    return best_arch

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

def plot_results(y_true, y_pred, costs):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Neural Network: Actual vs Predicted')
    plt.grid(True)
    
    residuals = y_true - y_pred
    plt.subplot(1, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(costs)
    plt.xlabel('Epoch')
    plt.ylabel('Cost (MSE)')
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('results/neural_network', exist_ok=True)
    plt.savefig('results/neural_network/neural_network_results.png', dpi=300)
    plt.close()

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("\n" + "="*60)
    print("NEURAL NETWORK - HOUSE PRICE PREDICTION")
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
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Normalize target
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    print("Target normalized for training stability")
    
    # Tune architecture
    best_arch = tune_architecture(
        X_train_scaled, y_train_norm.values, X_val_scaled, y_val_norm.values, learning_rate=0.01
    )
    
    # Train final model
    arch_str = ' -> '.join(map(str, best_arch))
    print(f"\nTraining final model ({arch_str})...")
    print(f"  Epochs: 1000")
    print(f"  Learning rate: 0.01")
    
    parameters, costs = train_neural_network(
        X_train_scaled, y_train_norm.values, best_arch, learning_rate=0.01, epochs=1000, verbose=False
    )
    print("✓ Training complete")
    
    # Predictions
    y_train_pred_norm = predict_neural_network(X_train_scaled, parameters)
    y_test_pred_norm = predict_neural_network(X_test_scaled, parameters)
    
    # Denormalize
    y_train_pred = y_train_pred_norm * y_std + y_mean
    y_test_pred = y_test_pred_norm * y_std + y_mean
    
    # Evaluate
    train_metrics = evaluate_model(y_train.values, y_train_pred)
    test_metrics = evaluate_model(y_test.values, y_test_pred)
    
    print("\nResults:")
    print(f"  Train - MSE: ${train_metrics['mse']:,.0f}, RMSE: ${train_metrics['rmse']:,.2f}, MAE: ${train_metrics['mae']:,.2f}, R²: {train_metrics['r2']:.4f}")
    print(f"  Test  - MSE: ${test_metrics['mse']:,.0f}, RMSE: ${test_metrics['rmse']:,.2f}, MAE: ${test_metrics['mae']:,.2f}, R²: {test_metrics['r2']:.4f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_results(y_test.values, y_test_pred, costs)
    print("✓ Plots saved to results/neural_network/")
    
    # Save metrics
    results_df = pd.DataFrame({
        'Metric': ['Architecture', 'Learning_Rate', 'Epochs', 
                   'Train_MSE', 'Train_RMSE', 'Train_MAE', 
                   'Test_MSE', 'Test_RMSE', 'Test_MAE', 
                   'Train_R2', 'Test_R2'],
        'Value': [
            arch_str, 0.01, 1000,
            train_metrics['mse'], train_metrics['rmse'], train_metrics['mae'],
            test_metrics['mse'], test_metrics['rmse'], test_metrics['mae'],
            train_metrics['r2'], test_metrics['r2']
        ]
    })
    os.makedirs('results/neural_network', exist_ok=True)
    results_df.to_csv('results/neural_network/neural_network_metrics.csv', index=False)
    print("✓ Metrics saved to results/neural_network/neural_network_metrics.csv")
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()