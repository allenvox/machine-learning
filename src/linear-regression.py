import numpy as np
import matplotlib.pyplot as plt

def linear_regression_until_real_delta(x_train, y_train, real_slope=4, alpha=0.01, delta_threshold=0.1):
    """
    Trains a linear regression model using gradient descent until the delta between the learned slope
    and the real slope of the linear function is within a specified threshold.

    Parameters:
    - x_train: Training data inputs
    - y_train: Training data outputs
    - real_slope: The actual slope of the linear function we are trying to learn
    - alpha: Learning rate
    - delta_threshold: Threshold for the delta between learned and real slope

    Returns:
    - W: Learned weights (W_0 and W_1)
    """
    X_design = np.vstack([np.ones(x_train.shape), x_train]).T
    W = np.zeros(2)  # Initialize weights
    delta_slope = np.inf
    while delta_slope > delta_threshold:
        y_pred = X_design.dot(W)
        error = y_pred - y_train
        W_gradient = X_design.T.dot(error) / len(x_train)
        W = W - alpha * W_gradient
        delta_slope = np.abs(W[1] - real_slope)
    return W

# Function to calculate maximum error between predictions and actual values
def calculate_maximum_error(x_test, y_test_real, W):
    """
    Calculates the maximum error between predicted and actual y values on the test set.

    Parameters:
    - x_test: Test data inputs
    - y_test_real: Actual outputs based on the real linear function (without error)
    - W: Learned weights

    Returns:
    - max_error: Maximum error between predicted and actual y values
    """
    y_pred = W[0] + W[1] * x_test
    max_error = np.max(np.abs(y_pred - y_test_real))
    return max_error

# Generate dataset
np.random.seed(42) # For reproducibility
x_values = np.linspace(-10, 10, 100)
y_values_real = 4 * x_values + 5 # Real values without error (4x + 5)
min_dataset_err = 0.01
max_dataset_err = 4.0
error = np.random.uniform(min_dataset_err, max_dataset_err, size=y_values_real.shape)
y_values_with_error = y_values_real + error

# Split the dataset
split_index = 80
x_train, y_train = x_values[:split_index], y_values_with_error[:split_index]
x_test, y_test_with_error = x_values[split_index:], y_values_with_error[split_index:]
y_test_real = y_values_real[split_index:]  # Actual values without error for testing

# Train model until delta between learned weight and real slope is within threshold
W_learned_delta = linear_regression_until_real_delta(x_train, y_train)
print("Learn results:")
print("W0 =", W_learned_delta[0], ", W1 =", W_learned_delta[1], "\n")

# Evaluate model on test set again
max_error_delta = calculate_maximum_error(x_test, y_test_real, W_learned_delta)
print("Test results:")
print("Max error =", max_error_delta)

# Generate predictions using the learned weights for the entire dataset for visualization
y_pred_train = W_learned_delta[0] + W_learned_delta[1] * x_train
y_pred_test = W_learned_delta[0] + W_learned_delta[1] * x_test


# Plotting
plt.figure(figsize=(10, 6))

# Real linear function
plt.plot(x_values, y_values_real, label='Real Linear Function (4x + 5)', color='blue')

# Learned results on training data
plt.plot(x_train, y_pred_train, 'g--', label='Learned Results (Training)', linewidth=2)

# Predictions on test data
plt.plot(x_test, y_pred_test, 'r--', label='Predictions (Test)', linewidth=2)

# Highlight training and test data points
plt.scatter(x_train, y_train, color='green', s=10, alpha=0.5, label='Training Data')
plt.scatter(x_test, y_test_with_error, color='red', s=10, alpha=0.5, label='Test Data')

plt.title('Linear Regression Model Comparison')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.legend()
plt.grid(True)
plt.show()
