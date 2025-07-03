import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = np.load('src/perception_pkg/ann_model/dataset.npy.npz')['arr_0']  # shape (N, 5)
X = data[:, :4]  # First 4 columns = features
y = data[:, 4].astype(np.int64)  # Last column = label

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define ANN model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)  # 2 output classes: blue (0), yellow (1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize model, loss, optimizer
model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    preds = torch.argmax(test_outputs, dim=1)
    acc = (preds == y_test).sum().item() / len(y_test)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# Save model
torch.save(model.state_dict(), 'src/perception_pkg/ann_model/model_ann.pth')
print("💾 Saved model to model_ann.pth ✅")
