import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class PortfolioSVM(nn.Module):
    def __init__(self, n_assets):
        super(PortfolioSVM, self).__init__()
        self.weights = nn.Parameter(torch.rand(n_assets))  # Initialize weights randomly

    def forward(self, x):
        # x is the input features, i.e., returns and risk metrics
        return torch.sigmoid(torch.matmul(x, self.weights))  # Apply sigmoid to scale outputs between 0 and 1

# Example Data: Assuming 'returns' and 'risk' are columns in your data
data_path = 'path_to_your_data.xlsx'
df = pd.read_excel(data_path)
returns = df['returns'].values
risk = df['risk'].values

# Normalize data
returns_norm = (returns - np.mean(returns)) / np.std(returns)
risk_norm = (risk - np.mean(risk)) / np.std(risk)

# Create tensors from numpy arrays
inputs = torch.tensor(np.vstack((returns_norm, risk_norm)).T, dtype=torch.float32)

n_assets = len(returns)
model = PortfolioSVM(n_assets)

# Initialize weights based on normalized returns and risk
init_weights = torch.tensor(returns_norm / risk_norm, dtype=torch.float32)
model.weights = nn.Parameter(init_weights)

n_assets = len(returns)
model = PortfolioSVM(n_assets)

# Initialize weights based on normalized returns and risk
init_weights = torch.tensor(returns_norm / risk_norm, dtype=torch.float32)
model.weights = nn.Parameter(init_weights)

def loss_fn(outputs, targets):
    return torch.mean((outputs - targets) ** 2)  # Example loss

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Example simplistic training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, torch.ones(outputs.shape))  # Dummy target of maximizing weights
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

with torch.no_grad():
    optimized_weights = model(inputs)
    print("Optimized Weights:", optimized_weights.numpy())
