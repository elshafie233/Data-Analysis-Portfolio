import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# housing price prediction ml model


# ---------- data ----------
def load_data():
    data = fetch_california_housing()
    X = data.data      # NumPy array, shape (20640, 8)
    y = data.target    # NumPy array, shape (20640,)
    return X, y



def split_and_scale(X, y, seed=42):
    # 60% train, 20% validation, 20% test
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)

    # one mean and one std PER COLUMN, from the training data only
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)

# feature scaling: (X - mu) / sigma

    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
    X_test = (X_test - mu) / sigma
    return X_train, X_val, X_test, y_train, y_val, y_test

 
# ---------- model ----------
def predict(X, w, b):
    return np.dot(X, w) + b


def compute_cost(X, y, w, b):
    m = X.shape[0]
    y_hat = predict(X, w, b)
    cost = (1 / (2 * m)) * np.sum(np.square(y_hat - y))
    return cost

 
def compute_gradient(X, y, w, b):
    m = X.shape[0]
    y_hat = predict(X, w, b)
    errors = y_hat - y
    dw = (1 / m) * np.dot(X.T, errors)
    db = (1 / m) * np.sum(errors)
    return dw, db

 
def gradient_descent(X, y, alpha=0.1, iters=1000):
    w = np.zeros(X.shape[1])
    b = 0.0
    costs = []


    for i in range(iters):
        dw, db = compute_gradient(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
 
        if i % 100 == 0:
            costs.append(compute_cost(X, y, w, b))

    return w, b, costs
 
 
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
 
 
# ---------- run ----------
def main():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(X, y)
 
    w, b, costs = gradient_descent(X_train, y_train)
 
    y_val_pred = predict(X_val, w, b)
    print(f"Validation RMSE: {rmse(y_val, y_val_pred):.4f}")
    print(f"Validation R^2:  {r2_score(y_val, y_val_pred):.4f}")
 
    plt.plot(range(0, len(costs) * 100, 100), costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost over Iterations")
    plt.savefig("cost_plot.png")   # saved as a file (works in Codespaces)
    plt.show()

if __name__ == "__main__":
    main()