import numpy as np
import pandas as pd


def calculate_portfolio_stats(weights, returns, risk_free_rate=0.06098):
    """
    Calculate portfolio statistics including return, standard deviation, and Sharpe ratio.
    """
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns, weights)
    portfolio_return = np.mean(portfolio_returns) * 252
    portfolio_std_dev = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev if portfolio_std_dev != 0 else 0

    # Calculate downside deviation and Sortino ratio
    target = risk_free_rate / 252
    downside_returns = np.where(portfolio_returns < target, portfolio_returns - target, 0)
    downside_std_dev = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
    sortino_ratio = (portfolio_return - risk_free_rate) / downside_std_dev if downside_std_dev != 0 else 0

    # Calculate diversification
    individual_std_devs = np.std(returns, axis=0)
    weighted_std_dev = np.sum(weights * individual_std_devs)
    diversification = portfolio_std_dev / weighted_std_dev if weighted_std_dev != 0 else 0

    # Calculate stability (autocorrelation of returns)
    stability = pd.Series(portfolio_returns).autocorr()

    return {
        'return': portfolio_return,
        'std_dev': portfolio_std_dev,
        'sharpe_ratio': sharpe_ratio,
        'downside_std_dev': downside_std_dev,
        'sortino_ratio': sortino_ratio,
        'diversification': diversification,
        'stability': stability
    }


def simulated_annealing(returns, initial_temp, cooling_rate, risk_free_rate):
    num_assets = returns.shape[1]
    weights = np.random.dirichlet(np.ones(num_assets))
    stats = calculate_portfolio_stats(weights, returns, risk_free_rate)
    best_solution = stats
    temperature = initial_temp

    while temperature > 1e-5:
        new_weights = np.random.dirichlet(np.ones(num_assets))
        new_stats = calculate_portfolio_stats(new_weights, returns, risk_free_rate)

        delta_E = new_stats['sharpe_ratio'] - stats['sharpe_ratio']

        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / temperature):
            weights = new_weights
            stats = new_stats

        if stats['sharpe_ratio'] > best_solution['sharpe_ratio']:
            best_solution = stats

        temperature *= cooling_rate

    return best_solution

data = pd.read_excel('Book6.xlsx', skiprows=[0], index_col=0)  # Adjust for your actual data source
print(data)
returns = data # Convert prices to returns


initial_temp = 1000
cooling_rate = 0.995
risk_free_rate = 0.06098  # Example risk-free rate

best_solution = simulated_annealing(returns, initial_temp, cooling_rate, risk_free_rate)
print(best_solution)

