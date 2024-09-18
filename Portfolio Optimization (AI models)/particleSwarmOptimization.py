import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# Load the data
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Initialize particles
class Particle:
    def __init__(self, num_assets):
        self.position = np.random.dirichlet(np.ones(num_assets), size=1).flatten()  # Ensure weights sum to 1
        self.velocity = np.zeros(num_assets)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, config):
        inertia = config['inertia']
        cognitive = config['cognitive']
        social = config['social']

        r1, r2 = np.random.random(), np.random.random()
        cognitive_velocity = cognitive * r1 * (self.best_position - self.position)
        social_velocity = social * r2 * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive_velocity + social_velocity

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, 0, None)  # No negative weights
        self.position /= np.sum(self.position)  # Normalize

# Portfolio Performance
def portfolio_performance(weights, returns, cov_matrix):
    returns_annual = np.dot(weights, returns.mean()) * 252
    volatility_annual = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return volatility_annual, returns_annual

# PSO Algorithm
def pso(data, num_particles, num_iterations, config):
    num_assets = data.shape[1]
    cov_matrix = data.cov() * 252
    global_best_position = None
    global_best_value = float('inf')
    particles = [Particle(num_assets) for _ in range(num_particles)]

    for _ in range(num_iterations):
        for particle in particles:
            current_volatility, current_return = portfolio_performance(particle.position, data, cov_matrix)
            current_value = current_volatility  # Minimizing volatility

            if current_value < particle.best_value:
                particle.best_value = current_value
                particle.best_position = np.copy(particle.position)

            if current_value < global_best_value:
                global_best_value = current_value
                global_best_position = np.copy(particle.position)

        for particle in particles:
            particle.update_velocity(global_best_position, config)
            particle.update_position()

    return global_best_position, global_best_value

# Function to calculate downside deviation
def calculate_downside_std(returns, weights, risk_free_rate=0.007829):
    portfolio_returns = np.dot(returns, weights)
    downside_returns = portfolio_returns[portfolio_returns < risk_free_rate]
    if len(downside_returns) == 0:
        return 0
    return np.std(downside_returns) * np.sqrt(252)

# Function to calculate Sortino ratio
def calculate_sortino_ratio(portfolio_returns, annual_return, risk_free_rate=0.007829):
    downside_std = calculate_downside_std(portfolio_returns, best_position, risk_free_rate)
    if downside_std == 0:
        return np.inf
    return (annual_return - risk_free_rate) / downside_std

# Function to calculate mean diversification
def calculate_diversification(individual_std_devs, portfolio_std, weights):
    weighted_std_devs = np.dot(weights, individual_std_devs)
    return weighted_std_devs / portfolio_std

# Function to calculate maximum drawdown
def calculate_max_drawdown(portfolio_returns):
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak
    return drawdown.max()

# Function to calculate the modified Sharpe ratio with absolute values
def modified_sharpe_ratio_with_absolute_values(portfolio_return, portfolio_volatility, risk_free_rate):
    abs_portfolio_return = np.abs(portfolio_return - risk_free_rate)
    portfolio_ratio = (portfolio_return - risk_free_rate) / abs_portfolio_return

    if portfolio_volatility != 0:  # Avoid division by zero
        modified_sharpe = (portfolio_return - risk_free_rate) / (portfolio_volatility ** portfolio_ratio)
    else:
        modified_sharpe = np.inf  # Handle case where standard deviation is zero

    return modified_sharpe

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
    return np.mean(stability) if stability else 0

# Load returns and set parameters
data = load_data('top2.xlsx')  # Update the path to your file

# Split the data into training and testing sets
split_ratio = 0.625  # 67% training, 33% testing
split_index = int(len(data) * split_ratio)
training_data = data[:split_index]
testing_data = data[split_index:]

# Configuration for PSO
config = {
    'inertia': 0.5,
    'cognitive': 1.5,
    'social': 1.5
}
num_particles = 40
num_iterations = 100
risk_free_rate = 0.00

# Run PSO on training data
best_position, best_value = pso(training_data, num_particles, num_iterations, config)
best_volatility_training, best_return_training = portfolio_performance(best_position, training_data, training_data.cov() * 252)

# Compute metrics for training data
sharpe_ratio_training = (best_return_training - risk_free_rate) / best_volatility_training if best_volatility_training != 0 else None
downside_std_training = calculate_downside_std(training_data, best_position, risk_free_rate)
sortino_ratio_training = calculate_sortino_ratio(training_data, best_return_training, risk_free_rate)
individual_std_devs_training = np.sqrt(np.diag(training_data.cov() * 252))
mean_diversification_training = calculate_diversification(individual_std_devs_training, best_volatility_training, best_position)
max_drawdown_training = calculate_max_drawdown(np.dot(training_data, best_position))
modified_sharpe_training = modified_sharpe_ratio_with_absolute_values(best_return_training, best_volatility_training, risk_free_rate)

# Calculate portfolio weights over time for training data
portfolio_weights_training = portfolio_weights_over_time(best_position, training_data)

# Calculate the diversification measure for training data
diversification_measure_training = np.sum(portfolio_weights_training ** 2, axis=1).mean()

# Calculate the portfolio stability for training data
mean_stability_portfolio_training = calculate_portfolio_stability(portfolio_weights_training)

print("Training Data:")
print("Best Weights:", best_position)
print("Best Annual Return:", best_return_training)
print("Best Annual Volatility:", best_volatility_training)
print("Sharpe Ratio:", sharpe_ratio_training)
print("Mean Downside Standard Deviation:", downside_std_training)
print("Sortino Ratio:", sortino_ratio_training)
print("Mean Diversification:", mean_diversification_training)
print("Mean Stability (Max Drawdown):", max_drawdown_training)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_training)
print("Diversification Measure (Sum of Squared Weights):", diversification_measure_training)
print("Portfolio Stability (Successive Periods):", mean_stability_portfolio_training)

# Apply the same weights to the testing data
best_volatility_testing, best_return_testing = portfolio_performance(best_position, testing_data, testing_data.cov() * 252)

# Compute metrics for testing data
sharpe_ratio_testing = (best_return_testing - risk_free_rate) / best_volatility_testing if best_volatility_testing != 0 else None
downside_std_testing = calculate_downside_std(testing_data, best_position, risk_free_rate)
sortino_ratio_testing = calculate_sortino_ratio(testing_data, best_return_testing, risk_free_rate)
individual_std_devs_testing = np.sqrt(np.diag(testing_data.cov() * 252))
mean_diversification_testing = calculate_diversification(individual_std_devs_testing, best_volatility_testing, best_position)
max_drawdown_testing = calculate_max_drawdown(np.dot(testing_data, best_position))
modified_sharpe_testing = modified_sharpe_ratio_with_absolute_values(best_return_testing, best_volatility_testing, risk_free_rate)

# Calculate portfolio weights over time for testing data
portfolio_weights_testing = portfolio_weights_over_time(best_position, testing_data)

# Calculate the diversification measure for testing data
diversification_measure_testing = np.sum(portfolio_weights_testing ** 2, axis=1).mean()

# Calculate the portfolio stability for testing data
mean_stability_portfolio_testing = calculate_portfolio_stability(portfolio_weights_testing)

print("Testing Data:")
print("Best Weights:", best_position)
print("Best Annual Return:", best_return_testing)
print("Best Annual Volatility:", best_volatility_testing)
print("Sharpe Ratio:", sharpe_ratio_testing)
print("Mean Downside Standard Deviation:", downside_std_testing)
print("Sortino Ratio:", sortino_ratio_testing)
print("Mean Diversification:", mean_diversification_testing)
print("Mean Stability (Max Drawdown):", max_drawdown_testing)
print("Modified Sharpe Ratio (with Absolute Values):", modified_sharpe_testing)
print("Diversification Measure (Sum of Squared Weights):", diversification_measure_testing)
print("Portfolio Stability (Successive Periods):", mean_stability_portfolio_testing)