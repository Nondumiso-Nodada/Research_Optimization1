import numpy as np
import pandas as pd


# Load data
def load_data(filepath):
    data = pd.read_excel('esg6.xlsx')
    return data
risk_free_rate = 0.06098


# Portfolio performance
def portfolio_std(weights, returns, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


# Initialization of population with random weights
def initialize_population(num_individuals, num_assets):
    population = np.random.rand(num_individuals, num_assets)
    population /= np.sum(population, axis=1)[:, None]  # Normalize weights so they sum to 1
    return population


# Selection of the fittest individuals
def select_parents(population, std_devs, num_parents):
    parents_idx = np.argsort(std_devs)[:num_parents]
    return population[parents_idx]


# Crossover to produce offspring
def crossover(parents, num_offspring):
    offspring = np.zeros((num_offspring, parents.shape[1]))
    crossover_point = np.uint8(parents.shape[1] / 2)

    for k in range(num_offspring):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring


# Mutation to introduce genetic diversity
def mutate(offspring, mutation_rate):
    for i in range(offspring.shape[0]):
        for j in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                # Randomly add or subtract a small value to the weight
                change = np.random.normal(0, 0.1)
                offspring[i, j] += change
        # Normalize the weights after mutation
        offspring[i, :] /= np.sum(offspring[i, :])
    return offspring


# Genetic Algorithm for Portfolio Optimization
def genetic_algorithm_optimize_risk(filepath, num_generations, population_size, num_parents, mutation_rate):
    # Load the returns data
    returns = load_data(filepath)
    cov_matrix = returns.cov()

    # Initialize population
    population = initialize_population(population_size, len(returns.columns))

    for generation in range(num_generations):
        # Calculate standard deviation for each individual
        std_devs = np.array([portfolio_std(ind, returns.mean(), cov_matrix) for ind in population])

        # Select parents
        parents = select_parents(population, std_devs, num_parents)

        # Crossover to create offspring
        num_offspring = population_size - parents.shape[0]
        offspring = crossover(parents, num_offspring)

        # Mutation
        offspring = mutate(offspring, mutation_rate)

        # Create new generation
        population[:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring

        # Find the best solution in the current generation
        gen_best_idx = np.argmin(std_devs)
        gen_best_std_dev = std_devs[gen_best_idx]

        if generation % 10 == 0:  # Print every 10 generations
            print(f"Generation {generation}: Best Std Dev: {gen_best_std_dev}")

    # Find the best solution at the end
    std_devs = np.array([portfolio_std(ind, returns.mean(), cov_matrix) for ind in population])
    best_idx = np.argmin(std_devs)
    best_solution = population[best_idx]

    return best_solution, std_devs[best_idx]


# Configuration
num_generations = 100  # Number of generations
population_size = 50  # Population size
num_parents = 20  # Number of parents to select for mating
mutation_rate = 0.01  # Mutation rate

# Run the optimization
filepath = 'esg6.xlsx'  # Replace with the correct path to your Excel file
best_weights, best_risk = genetic_algorithm_optimize_risk(
    filepath, num_generations, population_size, num_parents, mutation_rate
)

print("Optimized Weights:", best_weights)
print("Minimized Portfolio Risk:", best_risk)

# Assuming `best_weights` is the output from the GA

# Function to calculate portfolio's annual expected return
def annual_expected_return(weights, mean_returns):
    return np.sum(mean_returns * weights)
data = load_data('esg6.xlsx')
returns = data
cov_matrix = returns.cov()

# Function to calculate downside standard deviation
def downside_std_deviation(returns, weights, risk_free_rate):
    portfolio_returns = np.dot(returns, weights)
    downside_returns = portfolio_returns[portfolio_returns < risk_free_rate]
    if len(downside_returns) == 0:
        return 0
    return np.std(downside_returns) * np.sqrt(252)

# Function to calculate the Sortino ratio
def sortino_ratio(annual_return, downside_std, risk_free_rate):
    excess_return = annual_return - risk_free_rate
    return excess_return / downside_std if downside_std != 0 else np.inf

# Function to calculate mean diversification
def mean_diversification(weights, individual_stds):
    weighted_individual_std = np.dot(weights, individual_stds)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return weighted_individual_std / portfolio_std

# Function to calculate maximum drawdown, which represents mean stability
def max_drawdown(cumulative_returns):
    high_water_mark = np.maximum.accumulate(cumulative_returns)
    drawdown = (high_water_mark - cumulative_returns) / high_water_mark
    return np.max(drawdown)

# Now let's calculate these metrics using the optimized weights
mean_returns = returns.mean() * 252
annual_ret = annual_expected_return(best_weights, mean_returns)
portfolio_std_dev = portfolio_std(best_weights, returns, cov_matrix)

# Continue calculating the Sharpe ratio using the annual expected return and portfolio standard deviation
# Sharpe ratio
sharpe_ratio = (annual_ret - risk_free_rate) / portfolio_std_dev


# Calculate the downside standard deviation for Sortino ratio
downside_std = downside_std_deviation(returns.values, best_weights, risk_free_rate)

# Calculate the Sortino ratio using the downside standard deviation
sortino = sortino_ratio(annual_ret, downside_std, risk_free_rate)

# Calculate mean diversification
individual_stds = np.sqrt(np.diag(cov_matrix) * 252)  # Annualized standard deviations of individual assets
diversification = mean_diversification(best_weights, individual_stds)

# Calculate cumulative returns to find the max drawdown
portfolio_cumulative_returns = (1 + np.dot(returns, best_weights)).cumprod()
stability = max_drawdown(portfolio_cumulative_returns)

# Print the calculated metrics
print(f"Optimized Weights: {best_weights}")
print(f"Annual Expected Return: {annual_ret}")
print(f"Annual Standard Deviation (Volatility): {portfolio_std_dev}")
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Downside Standard Deviation: {downside_std}")
print(f"Sortino Ratio: {sortino}")
print(f"Mean Diversification Ratio: {diversification}")
print(f"Mean Stability (Max Drawdown): {stability}")

