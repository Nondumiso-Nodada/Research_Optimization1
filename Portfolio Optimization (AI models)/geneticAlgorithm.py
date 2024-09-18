import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# Load data
def load_data(filepath):
    data = pd.read_excel(filepath)
    return data

risk_free_rate = 0.008125

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
def genetic_algorithm_optimize_risk(returns, num_generations, population_size, num_parents, mutation_rate):
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

# Function to calculate portfolio's annual expected return
def annual_expected_return(weights, mean_returns):
    return np.sum(mean_returns * weights)

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
def mean_diversification(weights, individual_stds, portfolio_std):
    weighted_individual_std = np.dot(weights, individual_stds)
    return weighted_individual_std / portfolio_std

# Function to calculate maximum drawdown, which represents mean stability
def max_drawdown(cumulative_returns):
    high_water_mark = np.maximum.accumulate(cumulative_returns)
    drawdown = (high_water_mark - cumulative_returns) / high_water_mark
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

# Calculate portfolio weights over time
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

# Load returns and set parameters
returns = load_data('top1.xlsx')  # Update the path to your file

# Split the data into training and testing sets
split_ratio = 0.625  # 62.5% training, 37.5% testing
split_index = int(len(returns) * split_ratio)
training_data = returns[:split_index]
testing_data = returns[split_index:]

# Configuration
num_generations = 100  # Number of generations
population_size = 40  # Population size
num_parents = 20  # Number of parents to select for mating
mutation_rate = 0.01  # Mutation rate

# Run the optimization on the training data
best_weights, best_risk = genetic_algorithm_optimize_risk(
    training_data, num_generations, population_size, num_parents, mutation_rate
)

# Calculate metrics for training data
mean_returns_training = training_data.mean() * 252
cov_matrix_training = training_data.cov() * 252

annual_ret_training = annual_expected_return(best_weights, mean_returns_training)
portfolio_std_dev_training = portfolio_std(best_weights, training_data, cov_matrix_training)
sharpe_ratio_training = (annual_ret_training - risk_free_rate) / portfolio_std_dev_training

# Calculate the modified Sharpe ratio with absolute values for training data
modified_sharpe_training = modified_sharpe_ratio_with_absolute_values(annual_ret_training, portfolio_std_dev_training, risk_free_rate)

# Calculate downside standard deviation for training data
downside_std_training = downside_std_deviation(training_data.values, best_weights, risk_free_rate)

# Calculate the Sortino ratio using the downside standard deviation for training data
sortino_training = sortino_ratio(annual_ret_training, downside_std_training, risk_free_rate)

# Calculate mean diversification for training data
individual_stds_training = np.sqrt(np.diag(cov_matrix_training))  # Annualized standard deviations of individual assets
diversification_training = mean_diversification(best_weights, individual_stds_training, portfolio_std_dev_training)

# Calculate cumulative returns to find the max drawdown for training data
portfolio_cumulative_returns_training = (1 + np.dot(training_data, best_weights)).cumprod()
stability_training = max_drawdown(portfolio_cumulative_returns_training)

# Calculate portfolio weights over time for training data
portfolio_weights_training = portfolio_weights_over_time(best_weights, training_data)

# Calculate the diversification measure for training data
diversification_measure_training = calculate_diversification_measure(portfolio_weights_training)

# Calculate the portfolio stability for training data
mean_stability_portfolio_training = calculate_portfolio_stability(portfolio_weights_training)

# Output results for training data
print("Training Data:")
print(f"Optimized Weights: {best_weights}")
print(f"Annual Expected Return: {annual_ret_training}")
print(f"Annual Standard Deviation (Volatility): {portfolio_std_dev_training}")
print(f"Sharpe_ratio:", sharpe_ratio_training)
print(f"Modified Sharpe Ratio (with Absolute Values): {modified_sharpe_training}")
print(f"Downside Standard Deviation: {downside_std_training}")
print(f"Sortino Ratio: {sortino_training}")
print(f"Mean Diversification Ratio: {diversification_training}")
print(f"Mean Stability (Max Drawdown): {stability_training}")
print(f"Diversification Measure (Sum of Squared Weights): {diversification_measure_training}")
print(f"Portfolio Stability (Successive Periods): {mean_stability_portfolio_training}")

# Calculate metrics for testing data using the same optimal weights
mean_returns_testing = testing_data.mean() * 252
cov_matrix_testing = testing_data.cov() * 252

annual_ret_testing = annual_expected_return(best_weights, mean_returns_testing)
portfolio_std_dev_testing = portfolio_std(best_weights, testing_data, cov_matrix_testing)

sharpe_ratio_testing = (annual_ret_testing - risk_free_rate) / portfolio_std_dev_testing

# Calculate the modified Sharpe ratio with absolute values for testing data
modified_sharpe_testing = modified_sharpe_ratio_with_absolute_values(annual_ret_testing, portfolio_std_dev_testing, risk_free_rate)

# Calculate downside standard deviation for testing data
downside_std_testing = downside_std_deviation(testing_data.values, best_weights, risk_free_rate)

# Calculate the Sortino ratio using the downside standard deviation for testing data
sortino_testing = sortino_ratio(annual_ret_testing, downside_std_testing, risk_free_rate)

# Calculate mean diversification for testing data
individual_stds_testing = np.sqrt(np.diag(cov_matrix_testing))  # Annualized standard deviations of individual assets
diversification_testing = mean_diversification(best_weights, individual_stds_testing, portfolio_std_dev_testing)

# Calculate cumulative returns to find the max drawdown for testing data
portfolio_cumulative_returns_testing = (1 + np.dot(testing_data, best_weights)).cumprod()
stability_testing = max_drawdown(portfolio_cumulative_returns_testing)

# Calculate portfolio weights over time for testing data
portfolio_weights_testing = portfolio_weights_over_time(best_weights, testing_data)

# Calculate the diversification measure for testing data
diversification_measure_testing = calculate_diversification_measure(portfolio_weights_testing)

# Calculate the portfolio stability for testing data
mean_stability_portfolio_testing = calculate_portfolio_stability(portfolio_weights_testing)

# Output results for testing data
print("\nTesting Data:")
print(f"Annual Expected Return: {annual_ret_testing}")
print(f"Annual Standard Deviation (Volatility): {portfolio_std_dev_testing}")
print(f"Sharpe_ratio:", sharpe_ratio_testing)
print(f"Modified Sharpe Ratio (with Absolute Values): {modified_sharpe_testing}")
print(f"Downside Standard Deviation: {downside_std_testing}")
print(f"Sortino Ratio: {sortino_testing}")
print(f"Mean Diversification Ratio: {diversification_testing}")
print(f"Mean Stability (Max Drawdown): {stability_testing}")
print(f"Diversification Measure (Sum of Squared Weights): {diversification_measure_testing}")
print(f"Portfolio Stability (Successive Periods): {mean_stability_portfolio_testing}")
