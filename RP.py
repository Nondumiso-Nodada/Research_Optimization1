import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Load the data
data = pd.read_excel('Invst2.xlsx')  # Make sure to use the correct path to your Excel file
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
risk_free_rate = 0.020837  # Assuming a risk-free rate

sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility


# Function to calculate the modified Sharpe ratio with absolute values
def modified_sharpe_ratio_with_absolute_values(portfolio_return, portfolio_volatility, risk_free_rate):
    abs_portfolio_return = np.abs(portfolio_return - risk_free_rate)
    portfolio_ratio = (portfolio_return - risk_free_rate) / abs_portfolio_return

    if portfolio_volatility != 0:  # Avoid division by zero
        modified_sharpe = (portfolio_return - risk_free_rate) / (portfolio_volatility ** portfolio_ratio)
    else:
        modified_sharpe = np.inf  # Handle case where standard deviation is zero

    return modified_sharpe


# Calculate the modified Sharpe ratio with absolute values
modified_sharpe = modified_sharpe_ratio_with_absolute_values(portfolio_return, portfolio_volatility, risk_free_rate)

# Display results
print("Annual Expected Returns:\n", annual_returns)
print("Portfolio Volatility:", portfolio_volatility)
print("Portfolio Return:", portfolio_return)
print("Sharpe Ratio:", sharpe_ratio)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe)

# Calculate downside standard deviation
downside_returns = data[data < 0].fillna(0)
mean_downside_std_dev = np.sqrt(np.dot(normalized_weights.T, np.dot(downside_returns.cov() * 252, normalized_weights)))

sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

# Sortino Ratio
sortino_ratio = (portfolio_return - risk_free_rate) / mean_downside_std_dev

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

print("Mean Downside Standard Deviation:", mean_downside_std_dev)
print("Sortino Ratio:", sortino_ratio)
print("Mean Diversification:", mean_diversification)
print("Mean Stability (Inverse of Max Drawdown):", mean_stability)
