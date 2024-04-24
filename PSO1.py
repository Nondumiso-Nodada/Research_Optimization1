import numpy as np
import pandas as pd

# Load the data
def load_data(file_path):
    data = pd.read_excel('esg6.xlsx')
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
    volatility_annual = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return volatility_annual, returns_annual

# PSO Algorithm
def pso(file_path, num_particles, num_iterations, config):
    data = load_data(file_path)
    num_assets = data.shape[1]
    cov_matrix = data.cov()
    global_best_position = None
    global_best_value = float('inf')
    particles = [Particle(num_assets) for _ in range(num_particles)]

    for _ in range(num_iterations):
        for particle in particles:
            current_volatility, current_return = portfolio_performance(particle.position, data, cov_matrix)
            current_value = current_volatility  # Minimizing volatility

            if current_value < particle.best_value:
                particle.best_value = current_value
                particle.best_position = particle.position

            if current_value < global_best_value:
                global_best_value = current_value
                global_best_position = particle.position

        for particle in particles:
            particle.update_velocity(global_best_position, config)
            particle.update_position()

    return global_best_position, global_best_value

# Configuration and Execution
config = {
    'inertia': 0.5,
    'cognitive': 1.5,
    'social': 1.5
}
num_particles = 30
num_iterations = 100
risk_free_rate = 0.06098

best_position, best_value = pso('esg6.xlsx', num_particles, num_iterations, config)
best_volatility, best_return = portfolio_performance(best_position, load_data('esg6.xlsx'), load_data('esg6.xlsx').cov())
sharpe_ratio = best_return - risk_free_rate / best_volatility if best_volatility != 0 else None

print("Best Weights:", best_position)
print("Best Annual Return:", best_return)
print("Best Annual Volatility:", best_volatility)
print("Sharpe Ratio:", sharpe_ratio)


# Assuming data is already loaded and best_position is known
data = load_data('esg6.xlsx')  # Ensure this path is correct
returns = data
cov_matrix = returns.cov()

# Calculate individual asset standard deviations
individual_std_devs = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)

# Calculate portfolio returns for downside deviation
portfolio_returns = np.dot(returns, best_position)

def calculate_downside_std(portfolio_returns):
    downside_returns = portfolio_returns[portfolio_returns < 0]
    if len(downside_returns) == 0:
        return 0
    return np.std(downside_returns) * np.sqrt(252)

def calculate_sortino_ratio(portfolio_returns, annual_return, risk_free_rate=0.06098):
    downside_std = calculate_downside_std(portfolio_returns)
    if downside_std == 0:
        return np.inf
    return (annual_return - risk_free_rate) / downside_std

def calculate_diversification(individual_std_devs, portfolio_std):
    weighted_std_devs = np.dot(best_position, individual_std_devs)
    return weighted_std_devs / portfolio_std

def calculate_max_drawdown(portfolio_returns):
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak
    return drawdown.max()

# Compute metrics
downside_std = calculate_downside_std(portfolio_returns)
sortino_ratio = calculate_sortino_ratio(portfolio_returns, best_return)
mean_diversification = calculate_diversification(individual_std_devs, best_volatility)
max_drawdown = calculate_max_drawdown(portfolio_returns)

print("Mean Downside Standard Deviation:", downside_std)
print("Sortino Ratio:", sortino_ratio)
print("Mean Diversification:", mean_diversification)
print("Mean Stability (Max Drawdown):", max_drawdown)
