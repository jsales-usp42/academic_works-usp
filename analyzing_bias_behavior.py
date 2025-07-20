# =====================================================================
# Script to analyze the behavior of the graphs according the variation
# of the bias
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt

# Function to determine de function f(x)
def fx(x, w, b):
    return 1 / (1 + np.exp(-(w*x + b)))

# Values of x for plotting
x_vals = np.linspace(-2, 20, 200)

# Weights and bias for analysis
params = [(1, -8), (1, -7), (1, -6), (1, -5), (1, -4)]

# Plotting graphs for each parameter
plt.figure(figsize=(10, 6))
for w, b in params:
    y_vals = fx(x_vals, w, b)
    plt.plot(x_vals, y_vals, label=f'w={w}, b={b}')
plt.title("Analyzing bias behavior")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()
