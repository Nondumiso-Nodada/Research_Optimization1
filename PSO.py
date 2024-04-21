import pandas as pd
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import numpy as np
from scipy.optimize import minimize


# Adjust the path to where your actual Excel file is located
data = pd.read_excel('Book6.xlsx', skiprows=[0], index_col=0)  # Adjust for your actual data source
print(data)
returns = data


# Parameters for PSO
num_particles = 40
num_assets = returns.shape[1]
iterations = 100
w = 0.729  # Inertia weight
c1, c2 = 1.49445, 1.49445  # Cognitive and social coefficients

# Initialize positions and velocities
positions = np.random.dirichlet(np.ones(num_assets), size=num_particles)  # Ensure weights sum to 1
velocities = np.zeros((num_particles, num_assets))

# Initialize personal and global bests
personal_best_positions = positions.copy()
personal_best_scores = np.array([float('inf')] * num_particles)
global_best_position = None
global_best_score = float('inf')


def objective_function(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return -sharpe_ratio  # Negative because we want to maximize Sharpe ratio


# Optimization loop
for i in range(iterations):
    for j in range(num_particles):
        score = objective_function(positions[j], returns)
        # Update personal best
        if score < personal_best_scores[j]:
            personal_best_scores[j] = score
            personal_best_positions[j] = positions[j]
        # Update global best
        if score < global_best_score:
            global_best_score = score
            global_best_position = positions[j]

    # Update velocities and positions
    for j in range(num_particles):
        velocities[j] = w * velocities[j] + \
                        c1 * np.random.rand() * (personal_best_positions[j] - positions[j]) + \
                        c2 * np.random.rand() * (global_best_position - positions[j])
        positions[j] += velocities[j]
        positions[j] = np.clip(positions[j], 0, 1)  # Ensure no negative weights
        positions[j] /= np.sum(positions[j])  # Normalize to 1

print("Best position (asset weights):", global_best_position)
print("Best achieved Sharpe ratio:", -global_best_score)

# Assuming 'returns' is your DataFrame of asset returns
# and 'global_best_position' contains your optimal weights from PSO

# Annual Expected Return
annual_expected_return = np.sum(returns.mean() * global_best_position) * 252

# Portfolio Variance and Standard Deviation
portfolio_variance = np.dot(global_best_position.T, np.dot(returns.cov() * 252, global_best_position))
portfolio_std_dev = np.sqrt(portfolio_variance)

# Mean Downside Standard Deviation
# Filter negative returns or returns below a certain threshold
threshold = returns.dot(global_best_position).mean()  # using mean as a threshold for downside risk
downside_returns = returns.dot(global_best_position)[returns.dot(global_best_position) < threshold]
mean_downside_std_dev = np.sqrt(np.mean(downside_returns ** 2))

# Sortino Ratio (assuming risk-free rate is 0 for simplicity)
sortino_ratio = annual_expected_return / mean_downside_std_dev

# Mean Diversification using Herfindahl-Hirschman Index (HHI)
hhi = np.sum(global_best_position ** 2)
mean_diversification = 1 / hhi  # Higher values indicate better diversification

# Mean Stability using standard deviation of rolling returns
rolling_returns = returns.dot(global_best_position).rolling(window=30).mean()  # 21 trading days for monthly windows
mean_stability = rolling_returns.std()

# Display results
print("Annual Expected Return:", annual_expected_return)
print("Portfolio Standard Deviation:", portfolio_std_dev)
print("Mean Downside Standard Deviation:", mean_downside_std_dev)
print("Sortino Ratio:", sortino_ratio)
print("Mean Diversification:", mean_diversification)
print("Mean Stability:", mean_stability)
