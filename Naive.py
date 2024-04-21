import pandas as pd
import numpy as np
from scipy.optimize import minimize


data = pd.read_excel('Book6.xlsx', skiprows=[0], index_col=0)  # Adjust for your actual data source
print(data)


average_returns = data.mean()
std_devs = data.std()

num_assets = len(data.columns)
weights = np.ones(num_assets) / num_assets


portfolio_return = np.dot(weights, average_returns) * 252
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(data.cov(), weights)))
risk_free_rate = 0.06098  # Define according to your context
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

def downside_deviation(returns, target=0):
    return np.sqrt(np.mean(np.minimum(0, returns - target) ** 2))

portfolio_returns = np.dot(data, weights) * 252
downside_risk = downside_deviation(portfolio_returns)
sortino_ratio = (portfolio_return - risk_free_rate) / downside_risk

weighted_volatility = np.dot(weights, std_devs)
mean_diversification = weighted_volatility - portfolio_volatility

rolling_std_dev = pd.Series(portfolio_returns).rolling(window=30).std()
mean_stability = 1 / rolling_std_dev.mean()


print("Portfolio Return:", portfolio_return)
print("Portfolio Volatility:", portfolio_volatility)
print("Sharpe Ratio:", sharpe_ratio)
print("Downside Risk:", downside_risk)
print("Sortino Ratio:", sortino_ratio)
print("Mean Diversification:", mean_diversification)
print("Mean Stability:", mean_stability)


