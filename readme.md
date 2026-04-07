# Social Justice Multi-Perspective Neural Network

## Overview
This project builds a neural network model that attempts to balance multiple social justice perspectives simultaneously.

It includes:
- Class
- Community
- Religion
- Ethnicity
- Gender
- Caste
- Socioeconomic status
- Sexual orientation
- Race
- Age
- Education level

The goal is to simulate how these dimensions interact and produce a **"justice balance score"**.

---

## Conceptual Model

This model treats social justice as a **multi-dimensional optimization problem**.

Each dimension contributes to:
- Structural inequality
- Representation imbalance
- Opportunity distribution

The neural network learns:
- Non-linear interactions between identity factors
- Emergent fairness patterns
- Weighted influence of each dimension

---

## Architecture

Input Layer (11 features)
↓
Dense Layer (64 neurons, ReLU)
↓
Dense Layer (32 neurons, ReLU)
↓
Output Layer (1 neuron → justice score)

Loss Function: Mean Squared Error  
Optimizer: Adam

---

## Dataset

Synthetic dataset generated with:
- Random normalized features (0–1)
- Weighted fairness function
- Noise to simulate real-world uncertainty

---

## Outputs & Visualizations

### 1. Training Loss Curve
- Shows convergence of fairness learning

### 2. Justice Score Distribution
- Displays how society distributes fairness outcomes

### 3. Feature Influence Plot
- Correlation between each factor and outcome
- Reveals dominant social drivers

### 4. Actual vs Predicted
- Measures model accuracy
- Indicates how well fairness is approximated

---

## Key Insights

### 1. Social Justice is Non-Linear
The model learns interactions such as:
- Caste × socioeconomic status
- Gender × education
- Religion × community

These interactions are not additive — they are emergent.

---

### 2. Structural Dominance Emerges
Even in synthetic data, some features dominate:
- Socioeconomic status often becomes strongest
- Caste and class show strong coupling
- Education acts as a moderating factor

---

### 3. Fairness is a Distribution, Not a Point
The histogram shows:
- Justice is not binary
- It exists as a spread across population
- Inequality appears as skew

---

### 4. Model as a "Social Simulator"
This system can be extended into:
- Policy simulation engine
- Bias detection framework
- Governance modeling tool
- Ethical AI fairness system

---

## How to Run

```bash
pip install torch matplotlib numpy
python main.py
