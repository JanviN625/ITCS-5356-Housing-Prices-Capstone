# IntroMLCapstone - House Price Prediction

Comparative analysis of classical and advanced ML algorithms for residential property price prediction using the Ames Housing dataset.

## Project Structure
```
IntroMLCapstone/
├── data/
│   ├── train.csv                    # Training data
│   ├── test.csv                     # Test data (for Kaggle submission)
│   └── data_description.txt         # Feature descriptions
│
├── results/
│   ├── linear/                      # Linear regression outputs
│   ├── polynomial/                  # Polynomial regression outputs
│   ├── neural_network/              # Neural network outputs
│   ├── xgboost_paper/               # XGBoost paper results
│   └── stacking_paper/              # Stacking ensemble results
│
├── linear_regression.py             # Classical Model #1
├── polynomial_regression.py         # Classical Model #2
├── neural_network.py                # Classical Model #3
├── xgboost_paper.py                 # Paper #1: XGBoost (Phan, 2024)
├── stacking_paper.py                # Paper #2: Stacking (Truong et al., 2020)
│
├── capstone_report.pdf              # Final technical report
└── README.md                        # This file
```

## Models Implemented

### Classical Algorithms (Learned in Class)
1. **Linear Regression** (`linear_regression.py`)
   - Ridge regularization (α=10)
   - Closed-form and gradient descent solutions
   - Test R² = 0.871

2. **Polynomial Regression** (`polynomial_regression.py`)
   - Degree 3 polynomial features
   - Top 15 feature selection by correlation
   - Test R² = 0.836

3. **Neural Network** (`neural_network.py`)
   - Multilayer perceptron (260→100→1)
   - ReLU activation, backpropagation
   - Test R² = 0.865

### Literature-Based Methods
4. **XGBoost** (`xgboost_paper.py`)
   - Based on: Phan, T.D. (2024) "An Optimal House Price Prediction Algorithm: XGBoost"
   - Grid search over 81 hyperparameter combinations
   - Test R² = 0.925 (**Best model**)

5. **Stacking Ensemble** (`stacking_paper.py`)
   - Based on: Truong et al. (2020) "Housing Price Prediction via Improved Machine Learning Techniques"
   - Base models: Random Forest, XGBoost, LightGBM
   - Meta-learner: Ridge regression
   - Test R² = 0.898

## Requirements
```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm
```

## Usage

Run each model independently:
```bash
# Classical models
python linear_regression.py
python polynomial_regression.py
python neural_network.py

# Paper implementations
python xgboost_paper.py
python stacking_paper.py
```

Results (metrics, plots) are saved to respective `results/` subdirectories.

## Dataset

**Ames Housing Dataset** from Kaggle's "House Prices: Advanced Regression Techniques"
- 1,460 residential property sales
- 79 explanatory variables
- Target: Sale price in USD

## References

1. Phan, T.D. (2024). "An Optimal House Price Prediction Algorithm: XGBoost." *Analytics*, 3(1), 30-45.
2. Truong, Q., Nguyen, M., Dang, H., & Do, B. (2020). "Housing Price Prediction via Improved Machine Learning Techniques." *Procedia Computer Science*, 174, 433-442.

## Author

Janvi Nandwani  
University of North Carolina at Charlotte  
ITCS 5356 - Introduction to Machine Learning
