import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(x_train, y_train, alpha=0.01, delta_threshold=0.01, max_iterations=1000):
    X_design = np.vstack([np.ones(x_train.shape), x_train]).T
    W = np.zeros(2)  # Initialize weights
    delta_slope = np.inf
    iterations = 0
    while delta_slope > delta_threshold and iterations < max_iterations:
        z = np.dot(X_design, W)
        y_pred = sigmoid(z)
        error = y_pred - y_train
        W_gradient = np.dot(X_design.T, error) / len(x_train)
        W -= alpha * W_gradient
        delta_slope = np.linalg.norm(W_gradient)
        iterations += 1
    return W

# Generate binary dataset for classification
np.random.seed(42)
x_values = np.linspace(-10, 10, 100)
y_values_real = (x_values > 0).astype(int)

# Split the dataset
split_index = 80
x_train, y_train = x_values[:split_index], y_values_real[:split_index]
x_test, y_test = x_values[split_index:], y_values_real[split_index:]

# Train logistic regression model
W_learned = logistic_regression(x_train, y_train)
print("Learned weights:", W_learned)

# Test dataset across the entire range of x_values
x_test_full = x_values
y_test_full = y_values_real  # Use real values for comparison

# Predictions on full test set
X_test_full_design = np.vstack([np.ones(x_test_full.shape), x_test_full]).T
z_test_full = np.dot(X_test_full_design, W_learned)
y_pred_test_full = sigmoid(z_test_full)

# Plotting
plt.figure(figsize=(10, 6))

# Real binary function
plt.scatter(x_values, y_values_real, label='Real Binary Function', color='blue')

# Predictions on full test data
plt.scatter(x_test_full, y_pred_test_full, label='Predictions (Full Test)', color='green', alpha=0.5)

plt.title('Logistic Regression Model (Full Test)')
plt.xlabel('X Value')
plt.ylabel('Predicted Probability')
plt.legend()
plt.grid(True)
plt.show()
