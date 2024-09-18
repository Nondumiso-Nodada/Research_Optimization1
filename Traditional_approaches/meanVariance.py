import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

# Load your data
data = pd.read_excel('top3.xlsx')  # Update this path to your actual file path
data.columns = data.columns.str.strip()
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities
data.dropna(inplace=True)  # Drop rows with any NaN values

# Split the data into training and testing sets
split_ratio = 0.625  # 62.5% training, 37.5% testing
split_index = int(len(data) * split_ratio)
training_data = data[:split_index]
testing_data = data[split_index:]

# Annualize the daily returns for training data
annual_returns_training = training_data.mean() * 252

# Calculate the covariance matrix, annualized for training data
cov_matrix_training = training_data.cov() * 252
if np.isinf(cov_matrix_training.values).any() or np.isnan(cov_matrix_training.values).any():
    raise ValueError("Covariance matrix contains NaN or infinite values.")

# Risk-free rate
risk_free_rate = 0.008015

# Number of assets
num_assets = len(training_data.columns)

# Functions for portfolio metrics
def portfolio_return(weights, annual_returns):
    return np.dot(weights, annual_returns)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def sortino_ratio(weights, annual_returns, cov_matrix):
    p_return = portfolio_return(weights, annual_returns)
    negative_returns = training_data[training_data < 0].fillna(0)
    dd = np.sqrt(np.dot(weights.T, np.dot(negative_returns.cov() * 252, weights)))
    return (p_return - risk_free_rate) / dd

def diversification_ratio(weights, cov_matrix):
    weighted_volatilities = np.dot(weights, np.sqrt(np.diag(cov_matrix)))
    portfolio_vol = portfolio_volatility(weights, cov_matrix)
    return weighted_volatilities / portfolio_vol

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for asset in range(num_assets))
initial_weights = np.array(num_assets * [1. / num_assets])

# Optimization to minimize volatility on training data
opt_results = minimize(portfolio_volatility, initial_weights, args=(cov_matrix_training,), method='SLSQP', bounds=bounds, constraints=constraints)
if not opt_results.success:
    raise BaseException(opt_results.message)

optimal_weights = opt_results.x

# Calculations for training data
optimal_volatility_training = portfolio_volatility(optimal_weights, cov_matrix_training)
optimal_return_training = portfolio_return(optimal_weights, annual_returns_training)
sharpe_ratio_training = (optimal_return_training - risk_free_rate) / optimal_volatility_training
sortino_ratio_training = sortino_ratio(optimal_weights, annual_returns_training, cov_matrix_training)
div_ratio_training = diversification_ratio(optimal_weights, cov_matrix_training)

print("Training Data:")
print("Optimal weights:", optimal_weights)
print("Minimum Portfolio Volatility:", optimal_volatility_training)
print("Annual Expected Return:", optimal_return_training)
print("Sharpe Ratio:", sharpe_ratio_training)
print("Sortino Ratio:", sortino_ratio_training)
print("Diversification Ratio:", div_ratio_training)

# Calculate metrics for testing data using the same optimal weights
annual_returns_testing = testing_data.mean() * 252
cov_matrix_testing = testing_data.cov() * 252

optimal_volatility_testing = portfolio_volatility(optimal_weights, cov_matrix_testing)
optimal_return_testing = portfolio_return(optimal_weights, annual_returns_testing)
sharpe_ratio_testing = (optimal_return_testing - risk_free_rate) / optimal_volatility_testing
sortino_ratio_testing = sortino_ratio(optimal_weights, annual_returns_testing, cov_matrix_testing)
div_ratio_testing = diversification_ratio(optimal_weights, cov_matrix_testing)

# Portfolio daily returns for testing data
portfolio_daily_returns_testing = np.sum(testing_data * optimal_weights, axis=1)

def downside_std_dev(weights, data):
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

# Calculate the modified Sharpe ratio for testing data
modified_sharpe_testing = modified_sharpe_ratio_with_absolute_values(optimal_return_testing, optimal_volatility_testing, risk_free_rate)

# Calculate downside standard deviation and mean stability ratio for testing data
mean_downside_deviation_testing = downside_std_dev(optimal_weights, testing_data)
mean_stability_ratio_testing = 1 / -max_drawdown(portfolio_daily_returns_testing)

# Calculate diversification measure for both training and testing data
def calculate_diversification(weights, data):
    squared_weights = np.sum(weights ** 2)
    total_periods = data.shape[0]
    return squared_weights * total_periods

# Calculate portfolio weights over time
def portfolio_weights_over_time(weights, data):
    cumulative_returns = (1 + data).cumprod()
    portfolio_values = cumulative_returns.sum(axis=1)
    return cumulative_returns.div(portfolio_values, axis=0).mul(weights, axis=1)

# Portfolio Stability Measure
def calculate_portfolio_stability(weights_df):
    stability = []
    for i in range(1, len(weights_df)):
        stability_score = np.sum((weights_df.iloc[i] - weights_df.iloc[i - 1]) ** 2)
        stability.append(stability_score)
    return stability

# Diversification measure for training data
portfolio_weights_training = portfolio_weights_over_time(optimal_weights, training_data)
diversification_measure_training = portfolio_weights_training.apply(lambda row: np.sum(row ** 2), axis=1).mean()

# Portfolio stability measure for training data
stability_scores_training = calculate_portfolio_stability(portfolio_weights_training)
mean_stability_portfolio_training = np.mean(stability_scores_training)

# Diversification measure for testing data
portfolio_weights_testing = portfolio_weights_over_time(optimal_weights, testing_data)
diversification_measure_testing = portfolio_weights_testing.apply(lambda row: np.sum(row ** 2), axis=1).mean()

# Portfolio stability measure for testing data
stability_scores_testing = calculate_portfolio_stability(portfolio_weights_testing)
mean_stability_portfolio_testing = np.mean(stability_scores_testing)

# Output results for testing data
print("\nTesting Data:")
print("Portfolio Expected Annual Return:", optimal_return_testing)
print("Portfolio Volatility:", optimal_volatility_testing)
print("Sharpe Ratio:", sharpe_ratio_testing)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_testing)
print("Mean Downside Standard Deviation:", mean_downside_deviation_testing)
print("Sortino Ratio:", sortino_ratio_testing)
print("Diversification Measure (Sum of Squared Weights):", diversification_measure_testing)
print("Portfolio Stability (Successive Periods):", mean_stability_portfolio_testing)
