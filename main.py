import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------
# 1. Synthetic Social Dataset
# -----------------------------
np.random.seed(42)
torch.manual_seed(42)

N = 1000

def generate_data(N):
    # Features: class, religion, caste, gender, income, education
    class_ = np.random.randint(0, 3, N)
    religion = np.random.randint(0, 4, N)
    caste = np.random.randint(0, 3, N)
    gender = np.random.randint(0, 2, N)
    income = np.random.rand(N)
    education = np.random.rand(N)

    X = np.stack([class_, religion, caste, gender, income, education], axis=1).astype(np.float32)

    # Target fairness signal (synthetic ground truth)
    # Lower fairness if disadvantaged combination
    fairness = (
        0.5 * income +
        0.3 * education -
        0.2 * caste -
        0.1 * class_
    )

    fairness = (fairness - fairness.min()) / (fairness.max() - fairness.min())

    return torch.tensor(X), torch.tensor(fairness).unsqueeze(1)

X, y = generate_data(N)

# -----------------------------
# 2. Neural Network Model
# -----------------------------
class SocialJusticeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 16),   # input layer
            nn.ReLU(),

            nn.Linear(16, 16),  # intersectionality layer
            nn.ReLU(),

            nn.Linear(16, 8),   # social feedback abstraction
            nn.ReLU(),

            nn.Linear(8, 3),    # outputs: fairness, inequality, representation
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = SocialJusticeNet()

# -----------------------------
# 3. Custom Loss Function
# -----------------------------
def social_loss(outputs, target):
    fairness = outputs[:, 0]
    inequality = outputs[:, 1]
    representation = outputs[:, 2]

    # Loss components
    loss_fairness = ((fairness - target.squeeze()) ** 2).mean()
    loss_inequality = inequality.mean()   # minimize inequality
    loss_representation = -representation.mean()  # maximize representation

    # Weighted combination
    loss = (
        1.0 * loss_fairness +
        0.5 * loss_inequality +
        0.5 * loss_representation
    )

    return loss, loss_fairness.item(), loss_inequality.item(), loss_representation.item()

# -----------------------------
# 4. Training
# -----------------------------
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 150

loss_history = []
fairness_history = []
inequality_history = []
representation_history = []

for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model(X)
    loss, lf, li, lr = social_loss(outputs, y)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    fairness_history.append(lf)
    inequality_history.append(li)
    representation_history.append(lr)

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

# -----------------------------
# 5. Inference
# -----------------------------
with torch.no_grad():
    preds = model(X)

fairness_pred = preds[:, 0].numpy()
inequality_pred = preds[:, 1].numpy()
representation_pred = preds[:, 2].numpy()

# -----------------------------
# 6. Visualization
# -----------------------------
plt.figure(figsize=(12, 8))

# Loss curve
plt.subplot(2, 2, 1)
plt.plot(loss_history)
plt.title("Total Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Fairness
plt.subplot(2, 2, 2)
plt.plot(fairness_history)
plt.title("Fairness Loss")
plt.xlabel("Epoch")

# Inequality
plt.subplot(2, 2, 3)
plt.plot(inequality_history)
plt.title("Inequality (Minimized)")
plt.xlabel("Epoch")

# Representation
plt.subplot(2, 2, 4)
plt.plot(representation_history)
plt.title("Representation (Maximized)")
plt.xlabel("Epoch")

plt.tight_layout()
plt.show()

# Scatter: fairness vs income
plt.figure()
plt.scatter(X[:, 4].numpy(), fairness_pred)
plt.xlabel("Income")
plt.ylabel("Predicted Fairness")
plt.title("Fairness vs Income")
plt.show()
