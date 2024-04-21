import numpy as np
import pandas as pd
from scipy.stats import norm


# Load historical price data from a CSV file
data = pd.read_excel('Book6.xlsx', skiprows=[0], index_col=0)  # Adjust for your actual data source
print(data)
returns = data


# Function to calculate portfolio performance metrics
def portfolio_performance(weights, returns, risk_free_rate=0.06098):
    # Calculate return and standard deviation
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

    # Sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std

    return portfolio_return, portfolio_std, sharpe_ratio


# Fitness function to maximize Sharpe ratio
def fitness(weights, returns):
    _, _, sharpe_ratio = portfolio_performance(weights, returns)
    return sharpe_ratio


# Initialize population
population_size = 100
num_assets = len(returns.columns)
population = np.random.dirichlet(np.ones(num_assets), size=population_size)

# Evolutionary Algorithm parameters
num_generations = 50
mutation_rate = 0.05
selection_size = 20  # Number of individuals to select for the next generation

# EA main loop
for generation in range(num_generations):
    # Evaluate fitness
    fitness_scores = np.array([fitness(individual, returns) for individual in population])

    # Selection
    selected_indices = np.argsort(fitness_scores)[-selection_size:]
    selected = population[selected_indices]

    # Crossover (recombination)
    children = []
    while len(children) < population_size:
        parents = selected[np.random.choice(range(selection_size), 2, replace=False)]
        cross_point = np.random.randint(num_assets)
        child = np.concatenate([parents[0][:cross_point], parents[1][cross_point:]])
        children.append(child)
    children = np.array(children)

    # Mutation
    for i in range(population_size):
        if np.random.rand() < mutation_rate:
            children[i] = np.random.dirichlet(np.ones(num_assets))

    # ... [previous code] ...

    # EA main loop
    for generation in range(num_generations):
        # Evaluate fitness
        fitness_scores = np.array([fitness(individual, returns) for individual in population])

        # Selection
        selected_indices = np.argsort(fitness_scores)[-selection_size:]
        selected = population[selected_indices]

        # Crossover (recombination)
        children = []
        while len(children) < population_size:
            parents = selected[np.random.choice(range(selection_size), 2, replace=False)]
            cross_point = np.random.randint(num_assets)
            child = np.concatenate([parents[0][:cross_point], parents[1][cross_point:]])
            children.append(child)
        children = np.array(children)

        # Mutation
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                children[i] = np.random.dirichlet(np.ones(num_assets))

        # Replace the old population with children
        population = children

        # Normalize portfolio weights to sum to 1
        population /= np.sum(population, axis=1)[:, np.newaxis]

    # Identify the best solution
    best_index = np.argmax(fitness_scores)
    optimal_weights = population[best_index]



def downside_deviation(returns, weights, risk_free_rate=0.06098, target=0):
    """
    Calculate the downside deviation (mean downside standard deviation)
    for the portfolio.
    """
    portfolio_returns = returns.dot(weights)
    excess_returns = portfolio_returns - risk_free_rate / 252
    downside_returns = np.where(excess_returns < target, excess_returns ** 2, 0)
    mean_downside_returns = downside_returns.mean()
    return np.sqrt(mean_downside_returns) * np.sqrt(252)

def sortino_ratio(returns, weights, risk_free_rate=0.06098, target=0):
    """
    Calculate the Sortino ratio for the portfolio.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    dd = downside_deviation(returns, weights, risk_free_rate, target)
    sortino = (portfolio_return - risk_free_rate) / dd
    return sortino

def diversification_ratio(weights, returns):
    """
    Calculate the mean diversification ratio for the portfolio.
    """
    # Individual asset volatilities
    volatilities = returns.std(axis=0).values
    # Portfolio volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) * np.sqrt(252)
    # Diversification ratio
    div_ratio = np.sum(weights * volatilities) / portfolio_volatility
    return div_ratio

def stability_of_timeseries(returns):
    """
    Calculate the mean stability of the portfolio returns time series.
    This is a simple measure using the autocorrelation of the returns.
    """
    if len(returns) < 2:
        return float('nan')
    return 1 - abs(returns.autocorr())

# Function to calculate all performance metrics for a given set of portfolio weights
def portfolio_metrics(weights, returns, risk_free_rate=0.06098):
    portfolio_return, portfolio_std, sharpe_ratio = portfolio_performance(weights, returns, risk_free_rate)
    down_std = downside_deviation(returns, weights, risk_free_rate)
    sortino = sortino_ratio(returns, weights, risk_free_rate)
    div_ratio = diversification_ratio(weights, returns)
    stability = stability_of_timeseries(returns.dot(weights))
    return portfolio_return, portfolio_std, sharpe_ratio, down_std, sortino, div_ratio, stability

# Example usage in the evolutionary algorithm context:
# After finding the best weights in the population
optimal_return, optimal_std, optimal_sharpe_ratio, optimal_down_std, optimal_sortino_ratio, optimal_div_ratio, optimal_stability = portfolio_metrics(optimal_weights, returns)

print("Optimal Portfolio Metrics:")
print("Return:", optimal_return)
print("Standard Deviation:", optimal_std)
print("Sharpe Ratio:", optimal_sharpe_ratio)
print("Downside Standard Deviation:", optimal_down_std)
print("Sortino Ratio:", optimal_sortino_ratio)
print("Diversification Ratio:", optimal_div_ratio)
print("Stability:", optimal_stability)
