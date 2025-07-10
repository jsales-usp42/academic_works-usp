import numpy as np
import matplotlib.pyplot as plt

def fx(x, w, b):
    return 1 / (1 + np.exp(-(w*x + b)))

x_vals = np.linspace(-2, 20, 200)

params = [(1, -8), (1, -7), (1, -6), (1, -5), (1, -4)]
plt.figure(figsize=(10, 6))

for w, b in params:
    y_vals = fx(x_vals, w, b)
    plt.plot(x_vals, y_vals, label=f'w={w}, b={b}')

plt.title("")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()
