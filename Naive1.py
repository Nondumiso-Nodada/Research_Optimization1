import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Load your data
data = pd.read_excel('Invst1.xlsx')  # Adjust this to your file path
data.columns = data.columns.str.strip()

# Split the data into training and testing sets
split_ratio = 0.7  # 70% training, 30% testing
split_index = int(len(data) * split_ratio)
training_data = data[:split_index]
testing_data = data[split_index:]

# Calculate daily and annual metrics for training data
training_daily_returns = training_data
annual_returns_training = training_daily_returns.mean() * 252
annual_std_dev_training = training_daily_returns.std() * np.sqrt(252)
cov_matrix_training = training_daily_returns.cov() * 252

# Apply equal weights
num_assets = len(training_daily_returns.columns)
equal_weights = np.array([1 / num_assets] * num_assets)

# Portfolio metrics for training data
portfolio_return_training = np.dot(equal_weights, annual_returns_training)
portfolio_volatility_training = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix_training, equal_weights)))

# Risk-free rate
risk_free_rate = 0.023458

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

# Mean Downside Standard Deviation for training data
downside_returns_training = training_daily_returns[training_daily_returns < 0].fillna(0)
mean_downside_std_dev_training = np.sqrt(np.dot(equal_weights.T, np.dot(downside_returns_training.cov() * 252, equal_weights)))

# Sortino Ratio for training data
sortino_ratio_training = (portfolio_return_training - risk_free_rate) / mean_downside_std_dev_training

# Mean Diversification for training data
individual_volatilities_training = np.sqrt(np.diag(cov_matrix_training))
weighted_volatilities_training = np.dot(equal_weights, individual_volatilities_training)
mean_diversification_training = weighted_volatilities_training / portfolio_volatility_training

# Mean Stability (using max drawdown) for training data
cumulative_returns_training = (1 + training_daily_returns).cumprod()
peak_training = cumulative_returns_training.cummax()
drawdown_training = (cumulative_returns_training - peak_training) / peak_training
max_drawdown_training = drawdown_training.min().min()  # Min across time and assets
mean_stability_training = 1 / abs(max_drawdown_training)  # Inverse of max drawdown

# Apply the same weights to the testing data
testing_daily_returns = testing_data
annual_returns_testing = testing_daily_returns.mean() * 252
annual_std_dev_testing = testing_daily_returns.std() * np.sqrt(252)
cov_matrix_testing = testing_daily_returns.cov() * 252

# Portfolio metrics for testing data
portfolio_return_testing = np.dot(equal_weights, annual_returns_testing)
portfolio_volatility_testing = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix_testing, equal_weights)))

# Sharpe Ratio
sharpe_ratio_testing = (portfolio_return_testing - risk_free_rate) / portfolio_volatility_testing

# Calculate the modified Sharpe ratio with absolute values for testing data
modified_sharpe_testing = modified_sharpe_ratio_with_absolute_values(portfolio_return_testing, portfolio_volatility_testing, risk_free_rate)

# Mean Downside Standard Deviation for testing data
downside_returns_testing = testing_daily_returns[testing_daily_returns < 0].fillna(0)
mean_downside_std_dev_testing = np.sqrt(np.dot(equal_weights.T, np.dot(downside_returns_testing.cov() * 252, equal_weights)))

# Sortino Ratio for testing data
sortino_ratio_testing = (portfolio_return_testing - risk_free_rate) / mean_downside_std_dev_testing

# Mean Diversification for testing data
individual_volatilities_testing = np.sqrt(np.diag(cov_matrix_testing))
weighted_volatilities_testing = np.dot(equal_weights, individual_volatilities_testing)
mean_diversification_testing = weighted_volatilities_testing / portfolio_volatility_testing

# Mean Stability (using max drawdown) for testing data
cumulative_returns_testing = (1 + testing_daily_returns).cumprod()
peak_testing = cumulative_returns_testing.cummax()
drawdown_testing = (cumulative_returns_testing - peak_testing) / peak_testing
max_drawdown_testing = drawdown_testing.min().min()  # Min across time and assets
mean_stability_testing = 1 / abs(max_drawdown_testing)  # Inverse of max drawdown

# Output results for training data
print("Training Data:")
print("Portfolio Expected Annual Return:", portfolio_return_training)
print("Portfolio Volatility:", portfolio_volatility_training)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_training)
print("Mean Downside Standard Deviation:", mean_downside_std_dev_training)
print("Sortino Ratio:", sortino_ratio_training)
print("Mean Diversification:", mean_diversification_training)
print("Mean Stability (Inverse of Max Drawdown):", mean_stability_training)

# Output results for testing data
print("\nTesting Data:")
print("Portfolio Expected Annual Return:", portfolio_return_testing)
print("Portfolio Volatility:", portfolio_volatility_testing)
print("Sharpe_ratio:", sharpe_ratio_testing)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_testing)
print("Mean Downside Standard Deviation:", mean_downside_std_dev_testing)
print("Sortino Ratio:", sortino_ratio_testing)
print("Mean Diversification:", mean_diversification_testing)
print("Mean Stability (Inverse of Max Drawdown):", mean_stability_testing)
