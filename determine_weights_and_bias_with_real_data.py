# ============================================================================
# Script to determine weights (w) and bias (b) using real data from
# UCI Machine Learning Repository
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assign values to each education level:
education_map = {
    "illiterate": 1,
    "basic.4y": 2,
    "basic.6y": 3,
    "basic.9y": 4,
    "high.school": 5,
    "professional.course": 6,
    "university.degree": 7
}

# Function to load and prepare the .csv data file: 
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, sep=';')
    df.columns = df.columns.str.strip()

    df = df[df["education"] != "unknown"]
    df["education_num"] = df["education"].map(education_map)
    df = df.dropna(subset=["education_num"])
    df["y_binary"] = df["y"].map({"no": 0, "yes": 1})

    x_raw = df[["age", "education_num"]].values
    y = df["y_binary"].values

    # Normalize the x values:
    x_mean = np.mean(x_raw, axis=0)
    x_std = np.std(x_raw, axis=0)
    x_norm = (x_raw - x_mean) / x_std

    return x_norm, y, x_mean, x_std

# Function to define the function f(x):
def fx(w, x, b):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))

# Function to compute the loss function:
def cost(w, x, b, y):
    f = fx(w, x, b)
    epsilon = 1e-15 
    f = np.clip(f, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(f) + (1 - y) * np.log(1 - f))

# Function to compute the gradient:
def gradients(w, x, b, y):
    m = x.shape[0]
    f = fx(w, x, b)
    dJ_dw = np.dot(x.T, (f - y)) / m
    dJ_db = np.sum(f - y) / m
    return dJ_dw, dJ_db

# Function to determine the weights and bias:
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
    return w, b, J_history, i

# Function to plot the convergence graph:
def plot_convergence(J_history):
    plt.figure()
    plt.plot(J_history)
    plt.xlabel("Iterações")
    plt.ylabel("Função custo J(w,b)")
    plt.title("")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

x, y, x_mean, x_std = load_and_prepare_data("bank-additional.csv")

w_final, b_final, J_history, i = search_w_b(x, y)

print("w final:", w_final)
print("b final:", b_final)
print("Iterações:", i)
print("Último custo J:", J_history[-1])

plot_convergence(J_history)
