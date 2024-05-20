import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Load the data
data = pd.read_excel('Invst2.xlsx')  # Make sure to use the correct path to your Excel file
data.columns = data.columns.str.strip()

# Handle missing data
data.dropna(inplace=True)

# Split the data into training and testing sets
split_ratio = 0.7  # 70% training, 30% testing
split_index = int(len(data) * split_ratio)
training_data = data[:split_index]
testing_data = data[split_index:]

# Assuming 252 trading days in a year
# Training data
annual_returns_training = training_data.mean() * 252
annual_std_dev_training = training_data.std() * np.sqrt(252)
cov_matrix_training = training_data.cov() * 252

# Risk parity approach: weights are inversely proportional to volatility
weights_training = 1 / annual_std_dev_training
normalized_weights_training = weights_training / weights_training.sum()

# Portfolio metrics calculations for training data
portfolio_volatility_training = np.sqrt(np.dot(normalized_weights_training.T, np.dot(cov_matrix_training, normalized_weights_training)))
portfolio_return_training = np.dot(normalized_weights_training, annual_returns_training)
risk_free_rate = 0.020837  # Assuming a risk-free rate

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
mean_downside_std_dev_training = np.sqrt(np.dot(normalized_weights_training.T, np.dot(downside_returns_training.cov() * 252, normalized_weights_training)))

# Sortino Ratio for training data
sortino_ratio_training = (portfolio_return_training - risk_free_rate) / mean_downside_std_dev_training

# Mean Diversification for training data
individual_volatilities_training = annual_std_dev_training
weighted_volatilities_training = np.dot(normalized_weights_training, individual_volatilities_training)
mean_diversification_training = weighted_volatilities_training / portfolio_volatility_training

# Calculate portfolio returns for training data
portfolio_daily_returns_training = (training_data * normalized_weights_training).sum(axis=1)

# Calculate cumulative returns for training data
cumulative_returns_training = (1 + portfolio_daily_returns_training).cumprod()

# Calculate the maximum drawdown for training data
peak_training = cumulative_returns_training.cummax()
drawdown_training = (cumulative_returns_training - peak_training) / peak_training
max_drawdown_training = drawdown_training.min()

# Mean Stability as the inverse of the maximum drawdown for training data
mean_stability_training = 1 / abs(max_drawdown_training)  # Take the absolute value of max drawdown

# Output results for training data
print("Training Data:")
print("Annual Expected Returns:\n", annual_returns_training)
print("Portfolio Volatility:", portfolio_volatility_training)
print("Portfolio Return:", portfolio_return_training)
print("Sharpe Ratio:", sharpe_ratio_training)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_training)
print("Mean Downside Standard Deviation:", mean_downside_std_dev_training)
print("Sortino Ratio:", sortino_ratio_training)
print("Mean Diversification:", mean_diversification_training)
print("Mean Stability (Inverse of Max Drawdown):", mean_stability_training)

# Apply the same weights to the testing data
annual_returns_testing = testing_data.mean() * 252
annual_std_dev_testing = testing_data.std() * np.sqrt(252)
cov_matrix_testing = testing_data.cov() * 252

# Portfolio metrics calculations for testing data using the same weights from training
portfolio_volatility_testing = np.sqrt(np.dot(normalized_weights_training.T, np.dot(cov_matrix_testing, normalized_weights_training)))
portfolio_return_testing = np.dot(normalized_weights_training, annual_returns_testing)

sharpe_ratio_testing = (portfolio_return_testing - risk_free_rate) / portfolio_volatility_testing

# Calculate the modified Sharpe ratio with absolute values for testing data
modified_sharpe_testing = modified_sharpe_ratio_with_absolute_values(portfolio_return_testing, portfolio_volatility_testing, risk_free_rate)

# Calculate downside standard deviation for testing data
downside_returns_testing = testing_data[testing_data < 0].fillna(0)
mean_downside_std_dev_testing = np.sqrt(np.dot(normalized_weights_training.T, np.dot(downside_returns_testing.cov() * 252, normalized_weights_training)))

# Sortino Ratio for testing data
sortino_ratio_testing = (portfolio_return_testing - risk_free_rate) / mean_downside_std_dev_testing

# Mean Diversification for testing data
individual_volatilities_testing = annual_std_dev_testing
weighted_volatilities_testing = np.dot(normalized_weights_training, individual_volatilities_testing)
mean_diversification_testing = weighted_volatilities_testing / portfolio_volatility_testing

# Calculate portfolio returns for testing data
portfolio_daily_returns_testing = (testing_data * normalized_weights_training).sum(axis=1)

# Calculate cumulative returns for testing data
cumulative_returns_testing = (1 + portfolio_daily_returns_testing).cumprod()

# Calculate the maximum drawdown for testing data
peak_testing = cumulative_returns_testing.cummax()
drawdown_testing = (cumulative_returns_testing - peak_testing) / peak_testing
max_drawdown_testing = drawdown_testing.min()

# Mean Stability as the inverse of the maximum drawdown for testing data
mean_stability_testing = 1 / abs(max_drawdown_testing)  # Take the absolute value of max drawdown

# Output results for testing data
print("\nTesting Data:")
print("Annual Expected Returns:\n", annual_returns_testing)
print("Portfolio Volatility:", portfolio_volatility_testing)
print("Portfolio Return:", portfolio_return_testing)
print("Sharpe_ratio;", sharpe_ratio_testing)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_testing)
print("Mean Downside Standard Deviation:", mean_downside_std_dev_testing)
print("Sortino Ratio:", sortino_ratio_testing)
print("Mean Diversification:", mean_diversification_testing)
print("Mean Stability (Inverse of Max Drawdown):", mean_stability_testing)
