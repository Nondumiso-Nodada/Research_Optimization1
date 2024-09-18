import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import math

# Load data
def load_data(filepath):
    data = pd.read_excel(filepath)
    returns = data
    return returns

# Calculate portfolio statistics
def portfolio_stats(weights, returns, cov_matrix, risk_free_rate=0.008125):
    annual_returns = np.dot(weights, returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (annual_returns - risk_free_rate) / portfolio_volatility
    return annual_returns, portfolio_volatility, sharpe_ratio

# Generate a random neighbor
def random_neighbor(weights):
    perturbation = np.random.normal(0, 0.1, len(weights))
    new_weights = weights + perturbation
    new_weights[new_weights < 0] = 0
    new_weights /= np.sum(new_weights)  # Normalize
    return new_weights

# Simulated Annealing Algorithm
def simulated_annealing(returns, initial_weights, Tmax, Tmin, alpha):
    current_weights = initial_weights
    current_cov_matrix = returns.cov()
    current_returns, current_volatility, _ = portfolio_stats(current_weights, returns, current_cov_matrix)
    best_weights = np.copy(current_weights)
    best_volatility = current_volatility

    T = Tmax
    while T > Tmin:
        new_weights = random_neighbor(current_weights)
        new_returns, new_volatility, _ = portfolio_stats(new_weights, returns, current_cov_matrix)
        delta_E = new_volatility - current_volatility

        if delta_E < 0 or np.random.rand() < math.exp(-delta_E / T):
            current_weights = new_weights
            current_volatility = new_volatility

            if current_volatility < best_volatility:
                best_weights = current_weights
                best_volatility = current_volatility

        T *= alpha

    return best_weights

# Function to calculate downside deviation
def calculate_downside_std(returns, weights, risk_free_rate=0.008125):
    portfolio_returns = np.dot(returns, weights)
    downside_returns = portfolio_returns[portfolio_returns < risk_free_rate]
    if downside_returns.size > 0:
        return np.std(downside_returns) * np.sqrt(252)
    return 0

# Function to calculate Sortino ratio
def calculate_sortino_ratio(returns, weights, annual_return, risk_free_rate=0.008125):
    downside_std = calculate_downside_std(returns, weights, risk_free_rate)
    if downside_std > 0:
        return (annual_return - risk_free_rate) / downside_std
    return float('inf')

# Function to calculate mean diversification
def calculate_diversification(weights, individual_vols, portfolio_vol):
    weighted_individual_vols = np.dot(weights, individual_vols)
    if portfolio_vol > 0:
        return weighted_individual_vols / portfolio_vol
    return float('inf')

# Function to calculate maximum drawdown
def calculate_max_drawdown(portfolio_returns):
    cumulative = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return np.max(drawdown)

# Function to calculate the modified Sharpe ratio with absolute values
def modified_sharpe_ratio_with_absolute_values(portfolio_return, portfolio_volatility, risk_free_rate):
    abs_portfolio_return = np.abs(portfolio_return - risk_free_rate)
    portfolio_ratio = (portfolio_return - risk_free_rate) / abs_portfolio_return

    if portfolio_volatility != 0:  # Avoid division by zero
        modified_sharpe = (portfolio_return - risk_free_rate) / (portfolio_volatility ** portfolio_ratio)
    else:
        modified_sharpe = np.inf  # Handle case where standard deviation is zero

    return modified_sharpe

# Load returns and set parameters
returns = load_data('Esti3.xlsx')  # Update the path to your file

# Split the data into training and testing sets
split_ratio = 0.625  # 67% training, 33% testing
split_index = int(len(returns) * split_ratio)
training_data = returns[:split_index]
testing_data = returns[split_index:]

# Initial weights and parameters for simulated annealing
initial_weights = np.array([1 / training_data.shape[1]] * training_data.shape[1])
Tmax = 1.0
Tmin = 0.001
alpha = 0.9
risk_free_rate = 0.008125

# Run simulated annealing on training data
optimal_weights = simulated_annealing(training_data, initial_weights, Tmax, Tmin, alpha)
optimal_returns_training, optimal_volatility_training, optimal_sharpe_ratio_training = portfolio_stats(optimal_weights, training_data, training_data.cov())

# Compute the modified Sharpe ratio with absolute values for training data
modified_sharpe_training = modified_sharpe_ratio_with_absolute_values(optimal_returns_training, optimal_volatility_training, risk_free_rate)

# Using training data and optimal_weights obtained from previous steps
portfolio_returns_training = np.dot(training_data.values, optimal_weights)

mean_downside_std_training = calculate_downside_std(training_data.values, optimal_weights)
sortino_ratio_training = calculate_sortino_ratio(training_data.values, optimal_weights, optimal_returns_training)
individual_vols_training = np.sqrt(np.diag(training_data.cov()) * 252)
mean_diversification_training = calculate_diversification(optimal_weights, individual_vols_training, optimal_volatility_training)
max_drawdown_training = calculate_max_drawdown(portfolio_returns_training)

# Calculate portfolio weights over time for training data
def portfolio_weights_over_time(weights, data):
    cumulative_returns = (1 + data).cumprod()
    portfolio_values = cumulative_returns.sum(axis=1)
    return cumulative_returns.div(portfolio_values, axis=0).mul(weights, axis=1)

# Calculate the diversification measure
def calculate_diversification_measure(weights_df):
    return np.sum(weights_df ** 2, axis=1).mean()

# Calculate the portfolio stability
def calculate_portfolio_stability(weights_df):
    stability = []
    for i in range(1, len(weights_df)):
        stability_score = np.sum((weights_df.iloc[i] - weights_df.iloc[i - 1]) ** 2)
        stability.append(stability_score)
    return np.mean(stability) if stability else 0

# Calculate portfolio weights over time for training data
portfolio_weights_training = portfolio_weights_over_time(optimal_weights, training_data)

# Calculate the diversification measure for training data
diversification_measure_training = calculate_diversification_measure(portfolio_weights_training)

# Calculate the portfolio stability for training data
mean_stability_portfolio_training = calculate_portfolio_stability(portfolio_weights_training)

# Output results for training data
print("Training Data:")
print("Optimal Weights:", optimal_weights)
print("Optimal Annual Return:", optimal_returns_training)
print("Optimal Annual Standard Deviation:", optimal_volatility_training)
print("Optimal Sharpe Ratio:", optimal_sharpe_ratio_training)
print("Mean Downside Standard Deviation:", mean_downside_std_training)
print("Sortino Ratio:", sortino_ratio_training)
print("Mean Diversification:", mean_diversification_training)
print("Mean Stability (Max Drawdown):", max_drawdown_training)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_training)
print("Diversification Measure (Sum of Squared Weights):", diversification_measure_training)
print("Portfolio Stability (Successive Periods):", mean_stability_portfolio_training)

# Apply the same weights to the testing data
optimal_returns_testing, optimal_volatility_testing, optimal_sharpe_ratio_testing = portfolio_stats(optimal_weights, testing_data, testing_data.cov())

# Compute the modified Sharpe ratio with absolute values for testing data
modified_sharpe_testing = modified_sharpe_ratio_with_absolute_values(optimal_returns_testing, optimal_volatility_testing, risk_free_rate)

# Using testing data and optimal_weights obtained from previous steps
portfolio_returns_testing = np.dot(testing_data.values, optimal_weights)

mean_downside_std_testing = calculate_downside_std(testing_data.values, optimal_weights)
sortino_ratio_testing = calculate_sortino_ratio(testing_data.values, optimal_weights, optimal_returns_testing)
individual_vols_testing = np.sqrt(np.diag(testing_data.cov()) * 252)
mean_diversification_testing = calculate_diversification(optimal_weights, individual_vols_testing, optimal_volatility_testing)
max_drawdown_testing = calculate_max_drawdown(portfolio_returns_testing)

# Calculate portfolio weights over time for testing data
portfolio_weights_testing = portfolio_weights_over_time(optimal_weights, testing_data)

# Calculate the diversification measure for testing data
diversification_measure_testing = calculate_diversification_measure(portfolio_weights_testing)

# Calculate the portfolio stability for testing data
mean_stability_portfolio_testing = calculate_portfolio_stability(portfolio_weights_testing)

# Output results for testing data
print("\nTesting Data:")
print("Optimal Annual Return:", optimal_returns_testing)
print("Optimal Annual Standard Deviation:", optimal_volatility_testing)
print("Optimal Sharpe Ratio:", optimal_sharpe_ratio_testing)
print("Mean Downside Standard Deviation:", mean_downside_std_testing)
print("Sortino Ratio:", sortino_ratio_testing)
print("Mean Diversification:", mean_diversification_testing)
print("Mean Stability (Max Drawdown):", max_drawdown_testing)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_testing)
print("Diversification Measure (Sum of Squared Weights):", diversification_measure_testing)
print("Portfolio Stability (Successive Periods):", mean_stability_portfolio_testing)
