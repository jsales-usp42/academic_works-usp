# ==============================================================================
# Algorithm to determine the weights (w) and bias (b) using the
# Adam Adaptive Moment Estimation method
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assigning values from 1 to 7 according the education level
education_assigning_value = {
    "illiterate": 1,
    "basic.4y": 2,
    "basic.6y": 3,
    "basic.9y": 4,
    "high.school": 5,
    "professional.course": 6,
    "university.degree": 7
}

# Function to prepare the data that to be loaded:
def load_and_prepare_data(csv_path):
    # Load the .csv file:
    df = pd.read_csv(csv_path, sep = ';')
    # Keep only rows where "education" is different from "unknown":
    df = df[df["education"] != "unknown"]
    # Assigning values to education according "education_assigning_value":
    df["education_values"] = df["education"].map(education_assigning_value)
    # Assign values 0 and 1 to the output variable:
    df["y_0_1"] = df["y"].map({"no": 0, "yes": 1})
    # Select x and y data from DataFrame to be used for training:
    x = df[["age", "education_values"]].values
    y = df["y_0_1"].values
    # Normalize the x data:
    x_norm = (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
    return x_norm, y

# Function to determine the function f(x):
def fx(w, x, b):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))

# Function to compute the loss function:
def cost(w, x, b, y):
    f = fx(w, x, b)
    # Ensure that f(x) always stays beetwen 0 and 1 to avoid numeric issues:
    f = np.clip(f, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(f) + (1- y) * np.log(1 - f))

# Function to determine weights (w) and bias (b):
def search_w_b(x, y, cycles=10, epsilon=1e-6, alpha=0.1, beta1=0.9, beta2=0.999, epsilon_adam=1e-8):
    n = x.shape[1]
    w = np.zeros(n)
    b = 0.0
    m_w = np.zeros_like(w)
    v_w = np.zeros_like(w)
    m_b = 0.0
    v_b = 0.0
    t = 0
    J_history = []
    for j in range(cycles):
        indices = np.random.permutation(x.shape[0])
        for idx in indices:
            t += 1
            x_i = x[idx]
            y_i = y[idx]
            f_i = fx(w, x_i, b)
            dJ_dw = (f_i - y_i) * x_i
            dJ_db = (f_i - y_i)

            # Update moments (1st and 2nd order)
            m_w = beta1 * m_w + (1 - beta1) * dJ_dw
            v_w = beta2 * v_w + (1 - beta2) * (dJ_dw ** 2)
            m_b = beta1 * m_b + (1 - beta1) * dJ_db
            v_b = beta2 * v_b + (1 - beta2) * (dJ_db ** 2)

            # Correct bias:
            m_w_hat = m_w / (1 - beta1 ** t)
            v_w_hat = v_w / (1 - beta2 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)

            # Update parameters:
            w -= alpha * m_w_hat / (np.sqrt(v_w_hat) + epsilon_adam)
            b -= alpha * m_b_hat / (np.sqrt(v_b_hat) + epsilon_adam)

            # Total cost:
            J_calc = cost(w, x, b, y)
            J_history.append(J_calc)

            if len(J_history) > 1 and abs(J_history[-2] - J_calc) < epsilon:
                break
    return w, b, J_history, j + 1

# Function to evaluate the learning accuracy:
def evaluation(csv_path, w, b):
    df = pd.read_csv(csv_path, sep = ';')
    df = df[df["education"] != "unknown"]
    df["education_values"] = df["education"].map(education_assigning_value)
    df["y_0_1"] = df["y"].map({"no": 0, "yes": 1})
    x = df[["age", "education_values"]].values
    y_true = df["y_0_1"].values
    x_norm = (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
    y_pred_prob = 1 / (1 + np.exp(-(np.dot(x_norm, w) + b)))
    y_pred = (y_pred_prob >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y_true)
    print(f"Acurácia: {accuracy * 100:.2f}%")    
    
# Function to plot the convergence graph of the loss function:
def plot_convergence(J_history):
    plt.figure()
    plt.plot(J_history)
    plt.xlabel("Iterações")
    plt.ylabel("Função custo J(w, b)")
    plt.tight_layout()
    plt.show()

# Load .csv file for training:
x, y = load_and_prepare_data("bank-additional.csv")

# Determine weights (w) and bias (b):
w, b, J_final, i = search_w_b(x, y)

# Display the results:
print("w = ", w)
print("b = ", b)
print("Iterações = ", i)
print("Último custo J = ", J_final[-1])
evaluation("bank-additional-full.csv", w, b)
plot_convergence(J_final)
