
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================
# 1. Load and preprocess data
# ============================================
data = load_breast_cancer()
X = data.data          # 569 samples, 30 features
y = data.target        # 0 = malignant, 1 = benign

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert numpy arrays to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)   # shape (N, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# ============================================
# 2. Define the MLP architecture
# ============================================
class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        # Layer 1: 30 -> 16, then ReLU
        self.fc1 = nn.Linear(30, 16)
        self.relu = nn.ReLU()
        # Layer 2 (bottleneck): 16 -> 2, NO activation (linear)
        self.fc2 = nn.Linear(16, 2)
        # Output layer: 2 -> 1, then Sigmoid
        self.fc3 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))      # first hidden layer
        h2 = self.fc2(h1)                # 2D bottleneck (linear)
        out = self.sigmoid(self.fc3(h2)) # final probability
        return out, h2                   # return both prediction and 2D features

# ============================================
# 3. Training loop
# ============================================
model = CustomMLP()
criterion = nn.BCELoss()          # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 200
loss_history = []

print("\nTraining started...")
for epoch in range(epochs):
    # Forward pass: compute predictions and loss
    predictions, _ = model(X_train)
    loss = criterion(predictions, y_train)

    # Backward pass: compute gradients and update weights
    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # compute new gradients
    optimizer.step()        # update parameters

    loss_history.append(loss.item())

    # Print loss every 40 epochs
    if (epoch + 1) % 40 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Plot the training loss curve
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

# ============================================
# 4. Evaluate on test set
# ============================================
# Switch to evaluation mode (no gradient tracking)
model.eval()
with torch.no_grad():
    test_pred, test_hidden = model(X_test)
    # Convert probabilities to binary predictions (threshold 0.5)
    test_pred_binary = (test_pred >= 0.5).float()
    # Calculate accuracy
    correct = (test_pred_binary == y_test).sum().item()
    test_acc = correct / len(y_test)

print(f"\nTest Accuracy: {test_acc:.4f}")

# ============================================
# 5. Visualize the 2D bottleneck features
# ============================================
hidden_features = test_hidden.numpy()
labels = y_test.numpy().flatten()

plt.figure(figsize=(8, 6))
# Malignant (class 0) in red
plt.scatter(hidden_features[labels == 0, 0], hidden_features[labels == 0, 1],
            color='red', label='Malignant (0)', alpha=0.7, edgecolors='k')
# Benign (class 1) in blue
plt.scatter(hidden_features[labels == 1, 0], hidden_features[labels == 1, 1],
            color='blue', label='Benign (1)', alpha=0.7, edgecolors='k')

plt.title("2D Hidden Space Learned by MLP")
plt.xlabel("Hidden Neuron 1")
plt.ylabel("Hidden Neuron 2")
plt.legend()
plt.grid(True)
plt.show()