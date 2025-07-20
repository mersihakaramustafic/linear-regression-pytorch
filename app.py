import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

X = np.array([x for x in range(100)])
X = X.reshape(-1, 1)
y = 46 + 2 * X.flatten()

plt.scatter(X, y, label='Initial Data')
plt.title('Pre PyTorch')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

x_mean, x_std = X.mean(), X.std()
X_normalized = (X - x_mean) / x_std
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)

print(X_tensor.shape)

y_mean, y_std = y.mean(), y.std()
y_normalized = (y - y_mean) / y_std
y_tensor = torch.tensor(y_normalized, dtype=torch.float32)

print(y_tensor.shape)

class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x).squeeze(1)
    
in_features = 1
out_features = 1
model = LinearRegressionModel(in_features, out_features)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    # forward pass
    # outputs -> predictions
    outputs = model(X_tensor)
    # calculate loss
    # comparing predictions to real data
    loss = criterion(outputs, y_tensor)

    # backwardpass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.2f}')

new_x = 121
new_x_normalized = (new_x - x_mean) / x_std
new_x_tensor = torch.tensor(new_x_normalized, dtype=torch.float32).view(1, -1)

model.eval()
with torch.no_grad():
    prediction_normalized = model(new_x_tensor)

