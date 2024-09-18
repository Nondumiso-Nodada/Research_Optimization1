import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

# Load the data
data = pd.read_excel('top3.xlsx')  # Ensure the path to the file is correct
data.columns = data.columns.str.strip()

# Handle missing data
data.dropna(inplace=True)

# Split the data into training and testing sets
split_ratio = 0.625  # 67% training, 33% testing
split_index = int(len(data) * split_ratio)
training_data = data[:split_index]
testing_data = data[split_index:]

# Assuming 252 trading days in a year
# Calculate annualized returns and covariance matrix for the training data
annual_returns_training = training_data.mean() * 252
cov_matrix_training = training_data.cov() * 252

# Risk parity objective function
def risk_parity_objective(weights, cov_matrix):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    marginal_risk_contributions = np.dot(cov_matrix, weights) / portfolio_volatility
    risk_contributions = weights * marginal_risk_contributions
    risk_diff = risk_contributions - portfolio_volatility / len(weights)
    return np.sum(risk_diff ** 2)

# Constraints: sum of weights must be 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = [(0, 1) for _ in range(len(training_data.columns))]

# Initial guess
init_guess = np.ones(len(training_data.columns)) / len(training_data.columns)

# Optimization
result = minimize(risk_parity_objective, init_guess, args=(cov_matrix_training,), method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights
optimal_weights = result.x

# Portfolio metrics calculations for training data
portfolio_return_training = np.dot(optimal_weights, annual_returns_training)
portfolio_volatility_training = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix_training, optimal_weights)))
risk_free_rate = 0.008015  # Assuming a risk-free rate

sharpe_ratio_training = (portfolio_return_training - risk_free_rate) / portfolio_volatility_training

# Function to calculate the modified Sharpe ratio with absolute values
def modified_sharpe_ratio_with_absolute_values(portfolio_return, portfolio_volatility, risk_free_rate):
    abs_portfolio_return = np.abs(portfolio_return - risk_free_rate)
    portfolio_ratio = (portfolio_return - risk_free_rate) / abs_portfolio_return

    if portfolio_volatility != 0:  # Avoid division by zero
        modified_sharpe = (portfolio_return - risk_free_rate) / (portfolio_volatility ** portfolio_ratio)
    else:
        modified_sharpe = np.inf  # Handle case where standard deviation is zero

    return modified_sharpe

# Calculate the modified Sharpe ratio with absolute values for training data
modified_sharpe_training = modified_sharpe_ratio_with_absolute_values(portfolio_return_training, portfolio_volatility_training, risk_free_rate)

# Calculate downside standard deviation for training data
downside_returns_training = training_data[training_data < 0].fillna(0)
mean_downside_std_dev_training = np.sqrt(np.dot(optimal_weights.T, np.dot(downside_returns_training.cov() * 252, optimal_weights)))

# Sortino Ratio for training data
sortino_ratio_training = (portfolio_return_training - risk_free_rate) / mean_downside_std_dev_training

# Output results for training data
print("Training Data:")
print("Optimal Weights:", optimal_weights)
print("Annual Expected Return:", portfolio_return_training)
print("Portfolio Volatility:", portfolio_volatility_training)
print("Sharpe Ratio:", sharpe_ratio_training)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_training)
print("Mean Downside Standard Deviation:", mean_downside_std_dev_training)
print("Sortino Ratio:", sortino_ratio_training)

# Apply the same weights to the testing data
annual_returns_testing = testing_data.mean() * 252
cov_matrix_testing = testing_data.cov() * 252

# Portfolio metrics calculations for testing data using the same weights from training
portfolio_return_testing = np.dot(optimal_weights, annual_returns_testing)
portfolio_volatility_testing = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix_testing, optimal_weights)))

sharpe_ratio_testing = (portfolio_return_testing - risk_free_rate) / portfolio_volatility_testing

# Calculate the modified Sharpe ratio with absolute values for testing data
modified_sharpe_testing = modified_sharpe_ratio_with_absolute_values(portfolio_return_testing, portfolio_volatility_testing, risk_free_rate)

# Calculate downside standard deviation for testing data
downside_returns_testing = testing_data[testing_data < 0].fillna(0)
mean_downside_std_dev_testing = np.sqrt(np.dot(optimal_weights.T, np.dot(downside_returns_testing.cov() * 252, optimal_weights)))

# Sortino Ratio for testing data
sortino_ratio_testing = (portfolio_return_testing - risk_free_rate) / mean_downside_std_dev_testing

# Output results for testing data
print("\nTesting Data:")
print("Annual Expected Return:", portfolio_return_testing)
print("Portfolio Volatility:", portfolio_volatility_testing)
print("Sharpe Ratio:", sharpe_ratio_testing)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_testing)
print("Mean Downside Standard Deviation:", mean_downside_std_dev_testing)
print("Sortino Ratio:", sortino_ratio_testing)
