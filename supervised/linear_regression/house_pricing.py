import numpy as np
import matplotlib.pyplot as plt


def compute_model_output(x, w, b):
    """
    Computes the predicton of a linear model
    Args:
        x (ndarray (m,)): Data, m examples
        w, b (scalar)   : model parameters

    Returns:
        f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


# Training data
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

# m = number of training examples
m = x_train.shape[0]

w = 200
b = 100

f_wb = compute_model_output(x_train, w, b)

# Plot the data points
plt.plot(x_train, f_wb, label="Our Prediction", c="b")
plt.scatter(x_train, y_train, marker="x", c="r", label="Actual values")
plt.title("Housing Prices")
plt.ylabel("Price (in 1000a of dollar)")
plt.xlabel("Size (in 1000 sqft)")
plt.legend()
plt.show()
