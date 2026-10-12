import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LogisticRegressionFromScratch:
    """
    An elegant, vectorised implementation of L2-regularised Logistic Regression
    built entirely from scratch using NumPy.
    """
    def __init__(self, learning_rate=0.01, iterations=1000, lambda_reg=0.1):
        self.lr = learning_rate
        self.iterations = iterations
        self.lambda_reg = lambda_reg  
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        # Clip inputs to prevent computational overflow/underflow errors
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)

            # Calculate regularized log loss safely using clipping
            eps = 1e-15
            predictions = np.clip(predictions, eps, 1 - eps)
            loss = - (1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            l2_penalty = (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
            self.loss_history.append(loss + l2_penalty)

            # Compute regularized gradients
            dw = (1 / m) * np.dot(X.T, (predictions - y)) + (self.lambda_reg / m) * self.weights
            db = (1 / m) * np.sum(predictions - y)

            # Optimization step
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
  
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


def evaluate_model(model, X_test, y_test):
    """Computes standard binary classification metrics."""
    p = model.predict(X_test)
    return (
        accuracy_score(y_test, p),
        precision_score(y_test, p, zero_division=0),
        recall_score(y_test, p, zero_division=0),
        f1_score(y_test, p, zero_division=0)
    )


# Protected entry point: Essential for production code and clear portfolio style
if __name__ == "__main__":
    # 1. Pipeline Data Engineering
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale variables carefully strictly AFTER splitting to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Model Pipeline execution
    model = LogisticRegressionFromScratch(learning_rate=0.01, iterations=1000, lambda_reg=0.1)
    model.fit(X_train_scaled, y_train)
    
    # 3. Validation Assessment
    accuracy, precision, recall, f1 = evaluate_model(model, X_test_scaled, y_test)

    print("--- Professional Production Model Results (λ = 0.1) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
