import pandas as pd
import numpy as np
from scipy.optimize import minimize


# Load the data
data = pd.read_excel('Book3.xlsx', skiprows=[0], index_col=0)  # Adjust for your actual data source
print(data)

# Calculate expected returns
expected_returns = data.mean()
# print("Expected Returns:")
# print(expected_returns)

# Calculate covariance matrix
covariance_matrix = data.cov()
# print("covariance matrix:")
# print(covariance_matrix)


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


# Risk-free rate for Sharpe and Sortino Ratio calculations
risk_free_rate = 0.00


# Calculate Sharpe Ratio
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility



# print("Portfolio Expected Return:", portfolio_return)
print("Portfolio Volatility:", portfolio_volatility)
print("Sharpe Ratio:", sharpe_ratio)


# Calculate the correlation matrix
correlation_matrix = data.corr()

# Calculate the average of the off-diagonal elements in the correlation matrix (pairwise correlations)
# Using np.mean on the flattened upper triangle of the matrix minus 1 (to exclude the diagonal)


average_pairwise_correlation = (np.mean(correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)]))
mean_diversification = 1 - average_pairwise_correlation  # Diversification is better when correlations are low

print("Mean Diversification:", mean_diversification)


# Mean stability can be interpreted as low volatility of returns
# Calculate the standard deviation of portfolio returns (you might need to first calculate portfolio returns if not directly given)

# Assuming equal weights for simplicity, adjust as necessary for your portfolio weights
weights = np.ones(data.shape[1]) / data.shape[1]
portfolio_returns = data.dot(weights)

# Calculate the standard deviation of the portfolio returns
mean_stability = portfolio_returns.std()

print("Mean Stability:", mean_stability)






