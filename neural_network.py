"""
Neural Network - House Price Prediction
Capstone Project - Classical ML Algorithm #3

Manual NumPy implementation of Multi-Layer Perceptron (MLP).
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
# Activation Functions
# -----------------------------

def relu(Z):
    """ReLU activation function"""
    return np.maximum(0, Z)


def relu_derivative(Z):
    """Derivative of ReLU"""
    return (Z > 0).astype(float)


def identity(Z):
    """Identity activation for regression output"""
    return Z


def identity_derivative(Z):
    """Derivative of identity"""
    return np.ones_like(Z)

# -----------------------------
# Neural Network Implementation
# -----------------------------

def initialize_parameters(layer_dims):
    """
    Initialize weights and biases for neural network.
    
    Args:
        layer_dims: list of layer sizes [input, hidden1, hidden2, ..., output]
    
    Returns:
        parameters: dictionary containing W1, b1, W2, b2, ...
    """
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l-1], layer_dims[l]) * 0.01
        parameters[f'b{l}'] = np.zeros((1, layer_dims[l]))
    
    return parameters


def forward_propagation(X, parameters):
    """
    Forward pass through the network.
    
    Returns:
        cache: dictionary containing Z and A for each layer
        A_final: final output
    """
    cache = {}
    A = X
    L = len(parameters) // 2
    
    # Hidden layers with ReLU
    for l in range(1, L):
        Z = A @ parameters[f'W{l}'] + parameters[f'b{l}']
        A = relu(Z)
        cache[f'Z{l}'] = Z
        cache[f'A{l}'] = A
    
    # Output layer with identity (for regression)
    Z_out = A @ parameters[f'W{L}'] + parameters[f'b{L}']
    A_out = identity(Z_out)
    cache[f'Z{L}'] = Z_out
    cache[f'A{L}'] = A_out
    
    return cache, A_out


def backward_propagation(X, y, parameters, cache):
    """
    Backward pass to compute gradients.
    
    Returns:
        grads: dictionary containing gradients dW1, db1, dW2, db2, ...
    """
    grads = {}
    m = X.shape[0]
    L = len(parameters) // 2
    
    # Output layer gradient
    A_out = cache[f'A{L}']
    dZ = A_out - y.reshape(-1, 1)
    
    # Backpropagate through layers
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
    """Update parameters using gradient descent"""
    L = len(parameters) // 2
    
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    
    return parameters


def compute_cost(y_true, y_pred):
    """Compute MSE loss"""
    m = len(y_true)
    cost = (1/(2*m)) * np.sum((y_pred.flatten() - y_true) ** 2)
    return cost

# -----------------------------
# Model Training
# -----------------------------

def train_neural_network(X, y, layer_dims, learning_rate=0.01, epochs=1000, verbose=True):
    """
    Train neural network using gradient descent.
    
    Args:
        X: training features
        y: training target
        layer_dims: list of layer sizes
        learning_rate: learning rate
        epochs: number of training iterations
        verbose: print progress
    
    Returns:
        parameters: trained parameters
        costs: list of costs during training
    """
    parameters = initialize_parameters(layer_dims)
    costs = []
    
    for epoch in range(epochs):
        # Forward pass
        cache, A_out = forward_propagation(X, parameters)
        
        # Compute cost
        cost = compute_cost(y, A_out)
        costs.append(cost)
        
        # Backward pass
        grads = backward_propagation(X, y, parameters, cache)
        
        # Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print progress
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.2f}")
    
    return parameters, costs


def predict_neural_network(X, parameters):
    """Make predictions using trained network"""
    cache, A_out = forward_propagation(X, parameters)
    return A_out.flatten()

# -----------------------------
# Hyperparameter Tuning
# -----------------------------

def tune_architecture(X_train, y_train, X_val, y_val, learning_rate=0.01):
    """
    Find best network architecture.
    """
    print("\nTuning network architecture:")
    print("-" * 60)
    
    input_dim = X_train.shape[1]
    architectures = [
        [input_dim, 64, 1],
        [input_dim, 100, 1],
        [input_dim, 128, 1],
        [input_dim, 100, 50, 1],
        [input_dim, 128, 64, 1]
    ]
    
    best_score = float('inf')
    best_arch = None
    results = []
    
    for arch in architectures:
        arch_str = ' -> '.join(map(str, arch))
        
        # Train model
        params, costs = train_neural_network(
            X_train, y_train, arch, learning_rate, epochs=500, verbose=False
        )
        
        # Evaluate on validation set
        y_val_pred = predict_neural_network(X_val, params)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        results.append({
            'architecture': arch_str,
            'val_rmse': val_rmse
        })
        
        print(f"{arch_str:30s} -> Val RMSE: ${val_rmse:,.2f}")
        
        if val_rmse < best_score:
            best_score = val_rmse
            best_arch = arch
    
    print("-" * 60)
    best_arch_str = ' -> '.join(map(str, best_arch))
    print(f"Best architecture: {best_arch_str}")
    print(f"Best Val RMSE: ${best_score:,.2f}")
    
    # Save results
    pd.DataFrame(results).to_csv('neural_network_tuning.csv', index=False)
    
    return best_arch

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

def plot_results(y_true, y_pred, costs):
    """Plot predictions and training curve"""
    plt.figure(figsize=(15, 5))
    
    # Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], 
             [y_true.min(), y_true.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Neural Network: Actual vs Predicted')
    plt.grid(True)
    
    # Residuals
    residuals = y_true - y_pred
    plt.subplot(1, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    # Training curve
    plt.subplot(1, 3, 3)
    plt.plot(costs)
    plt.xlabel('Epoch')
    plt.ylabel('Cost (MSE)')
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('neural_network_results.png', dpi=300)
    print("Plot saved: neural_network_results.png")
    plt.show()

# -----------------------------
# Main Pipeline
# -----------------------------

def main():
    print("="*60)
    print("NEURAL NETWORK - HOUSE PRICE PREDICTION")
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
    
    # Scale features
    print("\n[Step 4] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Normalize target (helps with training stability)
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    print("Target normalized for training stability")
    
    # Tune architecture
    print("\n[Step 5] Tuning architecture...")
    best_arch = tune_architecture(
        X_train_scaled, y_train_norm.values,
        X_val_scaled, y_val_norm.values,
        learning_rate=0.01
    )
    
    # Train final model
    print(f"\n[Step 6] Training final model...")
    arch_str = ' -> '.join(map(str, best_arch))
    print(f"Architecture: {arch_str}")
    
    parameters, costs = train_neural_network(
        X_train_scaled, y_train_norm.values,
        best_arch, learning_rate=0.01, epochs=1000, verbose=True
    )
    
    # Make predictions
    print("\n[Step 7] Making predictions...")
    y_train_pred_norm = predict_neural_network(X_train_scaled, parameters)
    y_test_pred_norm = predict_neural_network(X_test_scaled, parameters)
    
    # Denormalize predictions
    y_train_pred = y_train_pred_norm * y_std + y_mean
    y_test_pred = y_test_pred_norm * y_std + y_mean
    
    # Evaluate
    print("\n[Step 8] Evaluating model...")
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
    print("\n[Step 9] Creating visualizations...")
    plot_results(y_test.values, y_test_pred, costs)
    
    # Save metrics
    print("\n[Step 10] Saving results...")
    results_df = pd.DataFrame({
        'Metric': ['Architecture', 'Learning_Rate', 'Epochs',
                   'Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2'],
        'Value': [
            arch_str,
            0.01,
            1000,
            train_metrics['rmse'],
            test_metrics['rmse'],
            train_metrics['r2'],
            test_metrics['r2']
        ]
    })
    results_df.to_csv('neural_network_metrics.csv', index=False)
    print("Metrics saved: neural_network_metrics.csv")
    
    print("\n" + "="*60)
    print("NEURAL NETWORK COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()