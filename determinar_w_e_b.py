import numpy as np
import matplotlib.pyplot as plt

def fx(w, x, b):
    fx = 1 / (1 + np.exp(-(np.dot(x, w) + b)))
    return fx

def cost(w, x, b, y):
    f = fx(w, x, b)
    J = -np.mean((1 - y)*(np.dot(x, w) + b) + np.log(1 + np.exp(-(np.dot(x, w) + b))))
    return J

def gradients(w, x, b, y):
    m = x.shape[0]
    f = fx(w, x, b)
    dJ_dw = np.dot(x.T, 1 - y - np.exp(-(np.dot(x, w) + b)) * f) / m
    dJ_db = np.sum(1 - y - np.exp(-(np.dot(x, w) + b)) * f) / m
    return dJ_dw, dJ_db

def search_w_b(x, y, alpha=0.1, iters=1000, epsilon=1e-6):
    n = x.shape[1]
    w = np.zeros(n)
    b = 0
    J_history = []
    for i in range(iters):
        dJ_dw, dJ_db = gradients(w, x, b, y)
        w -= alpha * dJ_dw
        b -= alpha * dJ_db

        J_calc = cost(w, x, b, y)
        J_history.append(J_calc)

        if i > 0 and abs(J_history[-2] - J_calc) < epsilon:
            break
    return w, b, J_calc, i

def plot_results(x, y, w, b):
    plt.figure()
    for label, marker, color in zip([0, 1], ['o', 's'], ['red', 'green']):
        plt.scatter(x[y == label, 0], x[y == label, 1], 
                    marker=marker, color=color, label=f"Probabilidade {label}")
    x1_vals = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
    x2_vals = -(w[0] * x1_vals + b) / w[1]
    plt.plot(x1_vals, x2_vals, 'k--', label='Fronteira de decisão')
    plt.xlabel("x1 (Renda)")
    plt.ylabel("x2 (Idade)")
    plt.legend()
    plt.title("")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Exemplo
x_example = np.array([
    [5, 34],
    [6, 24],
    [3, 53],
    [1, 55],
    [3, 44],
    [3, 63],
    [3, 53],
    [2, 70],
    [1, 72],
    [2, 45]
])
y_example = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 0])

w_final, b_final, cost_hist, i = search_w_b(x_example, y_example)

print("w: ", w_final)
print("b: ", b_final)
print("J(w, b): ", cost_hist)
print("i: ", i)
    
plot_results(x_example, y_example, w_final, b_final)
