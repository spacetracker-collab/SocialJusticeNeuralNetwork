import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------
# 1. Synthetic Dataset Generator
# -----------------------------
def generate_data(n_samples=2000):
    np.random.seed(42)

    # Features (normalized 0–1)
    data = {
        "class": np.random.rand(n_samples),
        "community": np.random.rand(n_samples),
        "religion": np.random.rand(n_samples),
        "ethnicity": np.random.rand(n_samples),
        "gender": np.random.rand(n_samples),
        "caste": np.random.rand(n_samples),
        "socioeconomic": np.random.rand(n_samples),
        "sexual_orientation": np.random.rand(n_samples),
        "race": np.random.rand(n_samples),
        "age": np.random.rand(n_samples),
        "education": np.random.rand(n_samples)
    }

    X = np.column_stack(list(data.values()))

    # Target: "justice balance score"
    # Simulated as weighted fairness function + noise
    weights = np.array([0.1,0.08,0.07,0.09,0.1,0.12,0.15,0.08,0.07,0.07,0.07])
    y = np.dot(X, weights) + 0.05*np.random.randn(n_samples)

    y = y.reshape(-1, 1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# -----------------------------
# 2. Neural Network Model
# -----------------------------
class SocialJusticeNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# 3. Training Function
# -----------------------------
def train_model(model, X, y, epochs=150):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss {loss.item():.4f}")

    return losses


# -----------------------------
# 4. Fairness Analysis
# -----------------------------
def fairness_analysis(model, X):
    with torch.no_grad():
        preds = model(X).numpy()

    # Analyze influence per feature (correlation proxy)
    influences = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i].numpy(), preds.flatten())[0, 1]
        influences.append(corr)

    return influences, preds


# -----------------------------
# 5. Main Execution
# -----------------------------
def main():
    X, y = generate_data()

    model = SocialJusticeNN(input_dim=X.shape[1])
    losses = train_model(model, X, y)

    influences, preds = fairness_analysis(model, X)

    feature_names = [
        "class","community","religion","ethnicity","gender",
        "caste","socioeconomic","sexual_orientation",
        "race","age","education"
    ]

    # -----------------------------
    # 6. Plotting (MULTI-SUBPLOT)
    # -----------------------------
    plt.figure(figsize=(14, 10))

    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Plot 2: Predicted Distribution
    plt.subplot(2, 2, 2)
    plt.hist(preds, bins=30)
    plt.title("Justice Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")

    # Plot 3: Feature Influence
    plt.subplot(2, 2, 3)
    plt.barh(feature_names, influences)
    plt.title("Feature Influence (Correlation)")
    plt.xlabel("Correlation")

    # Plot 4: Actual vs Predicted
    plt.subplot(2, 2, 4)
    plt.scatter(y.numpy(), preds, alpha=0.5)
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
