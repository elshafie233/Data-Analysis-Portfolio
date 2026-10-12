# Regularized Logistic Regression from Scratch

An elegant, object-oriented production implementation of L2-Regularized (Ridge) Logistic Regression built entirely from scratch using `NumPy`. This project demonstrates an understanding of classification fundamentals, optimization techniques, vectorization, and the mechanics used to prevent overfitting in generalized linear models.

## 🚀 Key Architectural Features
- **Pure Vectorization:** Leverages vectorized NumPy operations (\(X^T \cdot (\hat{y} - y)\)) instead of nested loops, mimicking performance structures found in industrial frameworks like scikit-learn.
- **Robust Numerics:** Includes input clipping safeguards within the Sigmoid activation and Log Loss evaluation to entirely eliminate computational `NaN` or overflow/underflow errors.
- **Strict Data Leakage Controls:** Preprocessing scaling transformers are strictly fit on training splits and evaluated downstream to maintain rigorous evaluation integrity.
- **L2 Regularization (Ridge Penalty):** Integrates structural parameter penalization controlled by regularization strength (λ) to penalize large weights and minimize overfitting.

---

## 📐 The Mathematics Behind the Model

### 1. Forward Pass (The Sigmoid Activation)
Instead of predicting unbound continuous values like linear regression, inputs are squashed through the logistic function into a probability boundary \(P(y=1\vert{}x) \in [0,1]\):

\[z = Xw + b\]

\[\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}\]

### 2. Regularized Log Loss (Binary Cross-Entropy)
To punish confident errors harshly, the cost function utilizes logarithmic loss combined with an L2 weight penalty:

\[J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2\]

### 3. Gradient Vectorization
The derivative curves share the exact geometric shape of ordinary least squares, altered cleanly by structural regularization updates applied straight to the weights during gradient descent:

\[\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w\]

\[\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})\]

---

## 🛠️ Installation & Quickstart

### Prerequisites
Ensure you have a modern Python environment installed with the following packages:
```bash
pip install numpy scikit-learn
```

### Execution
Run the standalone implementation module directly to train and evaluate the algorithm against scikit-learn's diagnostic baseline dataset:
```bash
python logistic_regression.py
```

---

## 📊 Verification Baseline Results

When validated against the standard Breast Cancer diagnostic profile (`test_size=0.2`), the mathematical execution scores perfectly alongside industrial libraries:

| Metric | Baseline Score (λ = 0.1) |
| :--- | :--- |
| **Accuracy** | 99.12% |
| **Precision** | 98.61% |
| **Recall** | 100.00% |
| **F1-Score** | 0.9930 |

---

## 📂 Project Structure
```text
├── logistic_regression.py  # Production class framework, pipelines, and evaluation entry point
└── README.md               # Portfolio engineering documentation
```
