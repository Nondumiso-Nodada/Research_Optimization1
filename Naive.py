import pandas as pd
import numpy as np

# Load your data
data = pd.read_excel('esg1.xlsx')  # Adjust this to your file path
data.columns = data.columns.str.strip()

# Calculate daily and annual metrics
daily_returns = data
annual_returns = daily_returns.mean() * 252
annual_std_dev = daily_returns.std() * np.sqrt(252)

# Apply equal weights
num_assets = len(daily_returns.columns)
equal_weights = np.array([1 / num_assets] * num_assets)

# Portfolio metrics
portfolio_return = np.dot(equal_weights, annual_returns)
portfolio_volatility = np.sqrt(np.dot(equal_weights.T, np.dot(daily_returns.cov() * 252, equal_weights)))

# Risk-free rate
risk_free_rate = 0.05666

# Sharpe Ratio
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

# Mean Downside Standard Deviation
downside_returns = daily_returns[daily_returns < 0].fillna(0)
mean_downside_std_dev = np.sqrt(np.dot(equal_weights.T, np.dot(downside_returns.cov() * 252, equal_weights)))

# Sortino Ratio
sortino_ratio = (portfolio_return - risk_free_rate) / mean_downside_std_dev

# Mean Diversification
individual_volatilities = np.sqrt(np.diag(daily_returns.cov() * 252))
weighted_volatilities = np.dot(equal_weights, individual_volatilities)
mean_diversification = weighted_volatilities / portfolio_volatility

# Mean Stability (using max drawdown)
cumulative_returns = (1 + daily_returns).cumprod()
peak = cumulative_returns.cummax()
drawdown = (cumulative_returns - peak) / peak
max_drawdown = drawdown.min().min()  # Min across time and assets
mean_stability = 1 / abs(max_drawdown)  # Inverse of max drawdown

# Output results
print("Portfolio Expected Annual Return:", portfolio_return)
print("Portfolio Volatility:", portfolio_volatility)
print("Sharpe Ratio:", sharpe_ratio)
print("Mean Downside Standard Deviation:", mean_downside_std_dev)
print("Sortino Ratio:", sortino_ratio)
print("Mean Diversification:", mean_diversification)
print("Mean Stability (Inverse of Max Drawdown):", mean_stability)
