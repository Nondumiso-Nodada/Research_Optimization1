import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import sem
from scipy.stats import skew, kurtosis


# Load your data
data = pd.read_excel('Invest3.xlsx')  # Update this path to your actual file path
data.columns = data.columns.str.strip()
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities
data.dropna(inplace=True)  # Drop rows with any NaN values

# Annualize the daily returns
annual_returns = data.mean() * 252

# Calculate the covariance matrix, annualized
cov_matrix = data.cov() * 252
if np.isinf(cov_matrix.values).any() or np.isnan(cov_matrix.values).any():
    raise ValueError("Covariance matrix contains NaN or infinite values.")

# Risk-free rate
risk_free_rate = 0.019806

# Number of assets
num_assets = len(data.columns)


# Functions for portfolio metrics
def portfolio_return(weights):
    return np.dot(weights, annual_returns)


def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def sortino_ratio(weights):
    p_return = portfolio_return(weights)
    negative_returns = data[data < 0].fillna(0)
    dd = np.sqrt(np.dot(weights.T, np.dot(negative_returns.cov() * 252, weights)))
    return (p_return - risk_free_rate) / dd


def diversification_ratio(weights):
    weighted_volatilities = np.dot(weights, np.sqrt(np.diag(cov_matrix)))
    portfolio_vol = portfolio_volatility(weights)
    return weighted_volatilities / portfolio_vol


# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for asset in range(num_assets))
initial_weights = np.array(num_assets * [1. / num_assets])

# Optimization to minimize volatility
opt_results = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
if not opt_results.success:
    raise BaseException(opt_results.message)

optimal_weights = opt_results.x

# Calculations
optimal_volatility = portfolio_volatility(optimal_weights)
optimal_return = portfolio_return(optimal_weights)
sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility
sortino_ratio_value = sortino_ratio(optimal_weights)
div_ratio = diversification_ratio(optimal_weights)

print("Optimal weights:", optimal_weights)
print("Minimum Portfolio Volatility:", optimal_volatility)
print("Annual Expected Return:", optimal_return)
print("Sharpe Ratio:", sharpe_ratio)
print("Sortino Ratio:", sortino_ratio_value)
print("Diversification Ratio:", div_ratio)


# Portfolio metrics functions
def portfolio_return(weights):
    return np.dot(weights, annual_returns)


def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def downside_std_dev(weights):
    negative_returns = data[data < 0].fillna(0)
    return np.sqrt(np.dot(weights.T, np.dot(negative_returns.cov() * 252, weights)))


def max_drawdown(returns_series):
    cumulative_returns = (1 + returns_series).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


# Calculate the modified Sharpe ratio with absolute values
def modified_sharpe_ratio_with_absolute_values(portfolio_return, portfolio_volatility, risk_free_rate):
    abs_portfolio_return = np.abs(portfolio_return - risk_free_rate)
    portfolio_ratio = (portfolio_return - risk_free_rate) / abs_portfolio_return

    if portfolio_volatility != 0:  # Avoid division by zero
        modified_sharpe = (portfolio_return - risk_free_rate) / (portfolio_volatility ** portfolio_ratio)
    else:
        modified_sharpe = np.inf  # Handle case where standard deviation is zero

    return modified_sharpe


# Calculate metrics
optimal_volatility = portfolio_volatility(optimal_weights)
optimal_return = portfolio_return(optimal_weights)
portfolio_daily_returns = np.sum(data * optimal_weights, axis=1)
mean_downside_deviation = downside_std_dev(optimal_weights)
mean_stability_ratio = 1 / -max_drawdown(portfolio_daily_returns)

# Calculate the modified Sharpe ratio
modified_sharpe = modified_sharpe_ratio_with_absolute_values(optimal_return, optimal_volatility, risk_free_rate)

print("Mean Downside Standard Deviation:", mean_downside_deviation)
print("Mean Stability Ratio (Inverse of Max Drawdown):", mean_stability_ratio)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe)
