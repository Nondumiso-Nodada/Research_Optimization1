import pandas as pd
import numpy as np
from scipy.optimize import minimize


data = pd.read_excel('Book6.xlsx', skiprows=[0], index_col=0)  # Adjust for your actual data source
print(data)

# Clean the column names by stripping any spaces
data.columns = data.columns.str.strip()

# Calculate the average returns for each asset
average_returns = data.mean()

# Calculate the standard deviation for each asset
std_devs = data.std()

# Calculate the covariance matrix of the asset returns
cov_matrix = data.cov()

# Risk-free rate assumption for Sharpe Ratio calculation
risk_free_rate = 0.06098

# Number of assets
num_assets = len(data.columns)

# Function to calculate portfolio returns, volatility, and Sharpe ratio
def portfolio_performance(weights, average_returns, cov_matrix, risk_free_rate):
    returns = np.dot(weights, average_returns) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / volatility
    return returns, volatility, sharpe_ratio

# Objective function to minimize (negative Sharpe Ratio)
def min_sharpe_ratio(weights, average_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, average_returns, cov_matrix, risk_free_rate)[2]

# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for each weight
bounds = tuple((0, 1) for asset in range(num_assets))

# Initial guess: equal weight for all assets
initial_guess = num_assets * [1. / num_assets]

# Optimization to find the optimal asset weights
opt_results = minimize(min_sharpe_ratio, initial_guess, args=(average_returns, cov_matrix, risk_free_rate),
                       method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = opt_results.x
portfolio_returns, portfolio_volatility, portfolio_sharpe_ratio = portfolio_performance(optimal_weights, average_returns, cov_matrix, risk_free_rate)

# Display results
print("Optimal Weights:\n", optimal_weights)
print("Expected Portfolio Return:", portfolio_returns)
print("Portfolio Volatility:", portfolio_volatility)
print("Portfolio Sharpe Ratio:", portfolio_sharpe_ratio)


# Function to calculate downside deviation
def downside_deviation(returns, target=0):
    return np.sqrt(np.mean(np.minimum(0, returns - target) ** 2))

# Extend portfolio performance function to include Sortino Ratio calculation
def extended_portfolio_performance(weights, average_returns, cov_matrix, data, risk_free_rate=0.06098, target=0):
    portfolio_returns = np.dot(data, weights) * 252
    mean_return = np.mean(portfolio_returns) * 252
    total_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    downside_risk = downside_deviation(portfolio_returns, target)
    sortino_ratio = (mean_return - risk_free_rate) / downside_risk if downside_risk > 0 else np.inf
    return mean_return, total_volatility, downside_risk, sortino_ratio

# Calculate the mean diversification (optional definition)
def mean_diversification(weights, std_devs):
    # Weighted average volatility
    weighted_volatility = np.dot(weights, std_devs)
    # Portfolio volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return weighted_volatility - portfolio_volatility

# Calculate diversification and Sortino ratio
mean_div = mean_diversification(optimal_weights, std_devs)
portfolio_mean_return, portfolio_volatility, portfolio_downside_risk, portfolio_sortino_ratio = extended_portfolio_performance(optimal_weights, average_returns, cov_matrix, data)



# Number of assets
num_assets = len(data.columns)

# Portfolio performance function including downside deviation
def portfolio_performance(weights, data, cov_matrix, risk_free_rate=0.06098):
    portfolio_returns = np.dot(data, weights) * 252
    mean_return = np.mean(portfolio_returns) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    downside_risk = np.sqrt(np.mean(np.minimum(0, portfolio_returns - risk_free_rate) ** 2))
    sortino_ratio = (mean_return - risk_free_rate) / downside_risk if downside_risk > 0 else np.inf
    return portfolio_returns, mean_return, volatility, downside_risk, sortino_ratio

# Minimize negative Sortino ratio
def min_sortino_ratio(weights, data, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, data, cov_matrix, risk_free_rate)[4]

# Optimization constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for asset in range(num_assets))
initial_guess = num_assets * [1. / num_assets]

# Optimize portfolio
opt_results = minimize(min_sortino_ratio, initial_guess, args=(data, cov_matrix, risk_free_rate),
                       method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_results.x

# Evaluate portfolio
portfolio_returns, mean_return, volatility, downside_risk, sortino_ratio = portfolio_performance(optimal_weights, data, cov_matrix)

# Calculate rolling standard deviation (e.g., monthly if data is daily)
rolling_std_dev = pd.Series(portfolio_returns).rolling(window=30).std()  # Adjust window size based on data frequency
mean_stability = 1 / rolling_std_dev.mean()  # Inverse of the mean of the rolling standard deviations


# Display additional results
print("Mean Diversification:", mean_div)
print("Downside Risk:", portfolio_downside_risk)
print("Sortino Ratio:", portfolio_sortino_ratio)
print("Mean Stability:", mean_stability)
