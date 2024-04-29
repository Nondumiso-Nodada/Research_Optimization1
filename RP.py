import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Load the data
data = pd.read_excel('esg6.xlsx')  # Make sure to use the correct path to your Excel file
data.columns = data.columns.str.strip()

# Handle missing data
data.dropna(inplace=True)

# Assuming 252 trading days in a year
annual_returns = data.mean() * 252
annual_std_dev = data.std() * np.sqrt(252)
cov_matrix = data.cov() * 252

# Risk parity approach: weights are inversely proportional to volatility
weights = 1 / annual_std_dev
normalized_weights = weights / weights.sum()

# Portfolio metrics calculations
portfolio_volatility = np.sqrt(np.dot(normalized_weights.T, np.dot(cov_matrix, normalized_weights)))
portfolio_return = np.dot(normalized_weights, annual_returns)
risk_free_rate = 0.010163 # Assuming a risk-free rate
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

# Display results
print("Normalized Weights:\n", normalized_weights)
print("Annual Expected Returns:\n", annual_returns)
print("Standard Deviations:\n", annual_std_dev)
print("Covariance Matrix:\n", cov_matrix)
print("Portfolio Volatility:", portfolio_volatility)
print("Portfolio Return:", portfolio_return)
print("Sharpe Ratio:", sharpe_ratio)

# Calculate daily and annual metrics
daily_returns = data
annual_returns = data.mean() * 252
annual_std_dev = data.std() * np.sqrt(252)
cov_matrix = data.cov() * 252

# Risk Parity: Inverse volatility weighting
weights = 1 / annual_std_dev
normalized_weights = weights / weights.sum()

# Portfolio metrics
portfolio_volatility = np.sqrt(np.dot(normalized_weights.T, np.dot(cov_matrix, normalized_weights)))
portfolio_return = np.dot(normalized_weights, annual_returns)

# Calculate downside standard deviation
downside_returns = data[data < 0]
downside_std_dev = downside_returns.std() * np.sqrt(252)
mean_downside_std_dev = np.sqrt(np.dot(normalized_weights.T, np.dot(downside_returns.cov() * 252, normalized_weights)))

# Sortino Ratio
expected_return = portfolio_return
sortino_ratio = (expected_return - risk_free_rate) / mean_downside_std_dev

# Mean Diversification
individual_volatilities = annual_std_dev
weighted_volatilities = np.dot(normalized_weights, individual_volatilities)
mean_diversification = weighted_volatilities / portfolio_volatility

# Calculate portfolio returns
portfolio_daily_returns = (data * normalized_weights).sum(axis=1)


# Calculate cumulative returns
cumulative_returns = (1 + portfolio_daily_returns).cumprod()

# Calculate the maximum drawdown
peak = cumulative_returns.cummax()
drawdown = (cumulative_returns - peak) / peak
max_drawdown = drawdown.min()

# Mean Stability as the inverse of the maximum drawdown
mean_stability = 1 / abs(max_drawdown)  # Take the absolute value of max drawdown


print("Mean Stability:", mean_stability)

# Display results
print("Mean Downside Standard Deviation:", mean_downside_std_dev)
print("Sortino Ratio:", sortino_ratio)
print("Mean Diversification:", mean_diversification)

data = pd.read_excel('esg5.xlsx')  # Make sure to use the correct path to your Excel file
data.columns = data.columns.str.strip()

# Handle missing data
data.dropna(inplace=True)

# Calculate daily returns
daily_returns = data

# Assuming 252 trading days in a year for annualization
annual_returns = daily_returns.mean() * 252
annual_std_dev = daily_returns.std() * np.sqrt(252)
cov_matrix = daily_returns.cov() * 252

# Risk parity approach: weights are inversely proportional to volatility
weights = 1 / annual_std_dev
normalized_weights = weights / weights.sum()

# Portfolio metrics calculations
portfolio_daily_returns = np.dot(daily_returns, normalized_weights)
portfolio_return = np.dot(normalized_weights, annual_returns)
portfolio_volatility = np.sqrt(np.dot(normalized_weights.T, np.dot(cov_matrix, normalized_weights)))

# Calculate skewness and kurtosis for the portfolio
portfolio_skewness = skew(portfolio_daily_returns)
portfolio_kurtosis = kurtosis(portfolio_daily_returns)

# Risk-free rate
risk_free_rate = 0.010163  # Update with the current risk-free rate

# Calculate the Modified Sharpe Ratio using the annualized figures
annual_excess_return = portfolio_return - risk_free_rate
portfolio_variance = portfolio_volatility ** 2
modified_sharpe_ratio = annual_excess_return / np.sqrt(portfolio_variance + (portfolio_skewness ** 2) + ((portfolio_kurtosis - 3) ** 2) / 4)

# Display results
print("Portfolio Volatility:", portfolio_volatility)
print("Portfolio Return:", portfolio_return)
print("Sharpe Ratio:", sharpe_ratio)
print("Modified Sharpe Ratio:", modified_sharpe_ratio)