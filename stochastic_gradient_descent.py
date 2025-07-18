import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Atribuindo valores de 1 a 7 de acordo com o nivel de escolaridade
education_assigning_value = {
    "illiterate": 1,
    "basic.4y": 2,
    "basic.6y": 3,
    "basic.9y": 4,
    "high.school": 5,
    "professional.course": 6,
    "university.degree": 7
}

# Funcao para preparar os dados que serao carregados:
def load_and_prepare_data(csv_path):
    # Carrega o arquivo .csv:
    df = pd.read_csv(csv_path, sep = ';')
    # Mantem apenas os dados quando "education" for diferente de "unknown":
    df = df[df["education"] != "unknown"]
    # Atribuir os valores à escolaridade conforme "education_assigning_value":
    df["education_values"] = df["education"].map(education_assigning_value)
    # Atribuir valores 0 e 1 na variavel de saida:
    df["y_0_1"] = df["y"].map({"no": 0, "yes": 1})
    # Determinar x e y os dados do DataFrame que serao usados no treinamento:
    x = df[["age", "education_values"]].values
    y = df["y_0_1"].values
    # Normalizar os dados de x:
    x_norm = (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
    return x_norm, y

# Funcao para determinar a funcao f(x):
def fx(w, x, b):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))

# Funcao para calcular a funcao custo:
def cost(w, x, b, y):
    f = fx(w, x, b)
    # Manter f(x) sempre entre 0 e 1 para evitar problemas numericos:
    f = np.clip(f, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(f) + (1- y) * np.log(1 - f))

# Funcao para determinar w e b:
def search_w_b(x, y, cycles = 50, alpha = 0.1, epsilon = 1e-6):
    n = x.shape[1]
    m = x.shape[0]
    w = np.zeros(n)
    b = 0
    J_history = []
    for j in range(cycles):
        indices = np.random.permutation(m)
        for idx in indices:
            x_i = x[idx]
            y_i = y[idx]
            f_i = fx(w, x_i, b)
            dJ_dw = (f_i - y_i) * x_i
            dJ_db = (f_i - y_i)
            w -= alpha * dJ_dw
            b -= alpha * dJ_db
            J_calc = cost(w, x, b, y)
            J_history.append(J_calc)
            if len(J_history) > 1 and abs(J_history[-2] - J_calc) < epsilon:
                break
    return w, b, J_history, j + 1

# Funcao para avaliar a acuracia do aprendizado:
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
    
# Funcao para plotar o grafico de convergencia da funcao custo
def plot_convergence(J_history):
    plt.figure()
    plt.plot(J_history)
    plt.xlabel("Iterações")
    plt.ylabel("Função custo J(w, b)")
    plt.tight_layout()
    plt.show()

# Carregando o arquivo para treinamento .csv:
x, y = load_and_prepare_data("bank-additional.csv")

# Determinando w e b:
w, b, J_final, i = search_w_b(x, y)

# Mostrando as saidas:
print("w = ", w)
print("b = ", b)
print("Iterações = ", i)
print("Último custo J = ", J_final[-1])
evaluation("bank-additional-full.csv", w, b)
plot_convergence(J_final)
