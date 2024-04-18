import pandas as pd
import numpy as np
from scipy.optimize import minimize

data = pd.read_excel('Book3.xlsx', skiprows=[0], index_col=0)  # Adjust for your actual data source
print(data)

threshold = 0
negative_returns = data[data < threshold].fillna(0)
print(negative_returns)


# Calculate covariance matrix
covariance_matrix = negative_returns.cov()
print("covariance matrix:")
print(covariance_matrix)

def risk_contribution(weights, covariance_matrix):
    # Total portfolio volatility
    portfolio_volatility = np.sqrt(weights @ covariance_matrix @ weights.T)
    # Marginal risk contribution
    marginal_risk_contribution = weights * (covariance_matrix @ weights) / portfolio_volatility
    # Risk contribution per asset
    risk_contribution = marginal_risk_contribution / portfolio_volatility
    return risk_contribution


def objective_function(weights, covariance_matrix):
    target_risk_contribution = np.ones(len(weights)) / len(weights)
    actual_risk_contribution = risk_contribution(weights, covariance_matrix)
    return np.sum((actual_risk_contribution - target_risk_contribution)**2)


num_assets = data.shape[1]
initial_weights = np.ones(num_assets) / num_assets  # Start with equal weights

constraints = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights must be 1
)
bounds = tuple((0, 1) for asset in range(num_assets))  # Weights between 0 and 1

optimized_result = minimize(
    objective_function,
    initial_weights,
    args=(covariance_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
optimized_weights = optimized_result.x


print("Optimized Weights for Risk Parity Portfolio:")
print(optimized_weights)



# Example expected returns for each asset
expected_returns = np.random.rand(40)  # You should replace this with your actual expected returns


# Example covariance matrix
covariance_matrix = np.random.rand(40, 40)  # Replace this with your actual covariance matrix
covariance_matrix = covariance_matrix @ covariance_matrix.T  # to ensure it's positive semi-definite


# Optimized weights from your input
weights = np.array(optimized_weights)  # Add all weights


# Calculate Portfolio Expected Return
portfolio_return = np.dot(weights, expected_returns)
print("portfolio_return")
print(portfolio_return)

# Calculate Portfolio Volatility (Standard Deviation)
portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)


print("portfolio_volatility")
print(portfolio_volatility)


# Risk-free rate for Sharpe and Sortino Ratio calculations
risk_free_rate = 0.00


# Assuming 'weights' is a numpy array containing your risk parity weights
# Calculate the downside deviation using matrix multiplication
downside_deviation = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

print("downside_deviation")
print(downside_deviation)


# Calculate the Sortino ratio
sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation


print("sortino_ratio")
print(sortino_ratio)
