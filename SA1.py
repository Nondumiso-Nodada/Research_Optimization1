import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import math


# Load data
def load_data(filepath):
    data = pd.read_excel('esg6.xlsx')
    returns = data
    return returns


# Calculate portfolio statistics
def portfolio_stats(weights, returns, cov_matrix, risk_free_rate=0.010163):
    annual_returns = np.dot(weights, returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (annual_returns - risk_free_rate) / portfolio_volatility
    return annual_returns, portfolio_volatility, sharpe_ratio
risk_free_rate = 0.010163

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


# Load returns and set parameters
returns = load_data('esg6.xlsx')  # Update the path to your file
initial_weights = np.array([1 / returns.shape[1]] * returns.shape[1])
Tmax = 1.0
Tmin = 0.001
alpha = 0.9

# Run simulated annealing
optimal_weights = simulated_annealing(returns, initial_weights, Tmax, Tmin, alpha)
optimal_returns, optimal_volatility, optimal_sharpe_ratio = portfolio_stats(optimal_weights, returns, returns.cov())



# Function to calculate downside deviation
def calculate_downside_std(returns, weights, risk_free_rate=0.010163):
    portfolio_returns = np.dot(returns, weights)
    downside_returns = portfolio_returns[portfolio_returns < risk_free_rate]
    if downside_returns.size > 0:
        return np.std(downside_returns) * np.sqrt(252)
    return 0

# Function to calculate Sortino ratio
def calculate_sortino_ratio(returns, weights, annual_return, risk_free_rate=0.010163):
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

# Using returns and optimal_weights obtained from previous steps
portfolio_returns = np.dot(returns.values, optimal_weights)

mean_downside_std = calculate_downside_std(returns.values, optimal_weights)
sortino_ratio = calculate_sortino_ratio(returns.values, optimal_weights, optimal_returns)
individual_vols = np.sqrt(np.diag(returns.cov()) * 252)
mean_diversification = calculate_diversification(optimal_weights, individual_vols, optimal_volatility)
max_drawdown = calculate_max_drawdown(portfolio_returns)


print("Optimal Weights:", optimal_weights)
print("Optimal Annual Return:", optimal_returns)
print("Optimal Annual Standard Deviation:", optimal_volatility)
print("Optimal Sharpe Ratio:", optimal_sharpe_ratio)
print("Mean Downside Standard Deviation:", mean_downside_std)
print("Sortino Ratio:", sortino_ratio)
print("Mean Diversification:", mean_diversification)
print("Mean Stability (Max Drawdown):", max_drawdown)


# Function to load data
def load_data(filepath):
    data = pd.read_excel('esg6.xlsx')
    return data


# Function to calculate portfolio statistics
def portfolio_stats(weights, returns, cov_matrix, risk_free_rate):
    annual_returns = np.dot(weights, returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return annual_returns, portfolio_volatility


# Generate a random neighbor
def random_neighbor(weights):
    perturbation = np.random.normal(0, 0.1, len(weights))
    new_weights = weights + perturbation
    new_weights = np.clip(new_weights, 0, None)
    new_weights /= np.sum(new_weights)
    return new_weights


# Simulated Annealing Algorithm
def simulated_annealing(returns, initial_weights, Tmax, Tmin, alpha, risk_free_rate):
    current_weights = initial_weights
    current_cov_matrix = returns.cov()
    current_returns, current_volatility = portfolio_stats(current_weights, returns, current_cov_matrix, risk_free_rate)
    best_weights = np.copy(current_weights)
    best_returns = current_returns
    best_volatility = current_volatility

    T = Tmax
    while T > Tmin:
        new_weights = random_neighbor(current_weights)
        new_returns, new_volatility = portfolio_stats(new_weights, returns, current_cov_matrix, risk_free_rate)
        delta_E = new_volatility - current_volatility

        if delta_E < 0 or np.random.rand() < math.exp(-delta_E / T):
            current_weights = new_weights
            current_volatility = new_volatility
            current_returns = new_returns

            if current_volatility < best_volatility:
                best_weights = current_weights
                best_returns = current_returns
                best_volatility = current_volatility

        T *= alpha

    # Calculate skewness and kurtosis for the best returns
    returns_series = np.dot(returns, best_weights)
    skewness = skew(returns_series)
    kurtosis_value = kurtosis(returns_series)

    # Calculate the Modified Sharpe Ratio
    modified_sharpe_ratio = (best_returns - risk_free_rate) / np.sqrt(
        best_volatility ** 2 + skewness ** 2 + (kurtosis_value-3) ** 2 / 4)
    return best_weights, modified_sharpe_ratio


# Load returns and set parameters
returns = load_data('esg6.xlsx')  # Update the path to your file
initial_weights = np.array([1 / returns.shape[1]] * returns.shape[1])
Tmax = 1.0
Tmin = 0.001
alpha = 0.9
risk_free_rate = 0.010163

# Run simulated annealing
optimal_weights, modified_sharpe = simulated_annealing(returns, initial_weights, Tmax, Tmin, alpha, risk_free_rate)

print("Optimal Weights:", optimal_weights)
print("Modified Sharpe Ratio:", modified_sharpe)