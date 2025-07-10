import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def add_bias_column(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def cost_function(theta, X, y):
    predictions = X @ theta
    error = y - predictions
    return np.sum(error ** 2)

def regressao_otimizacao(X, y, plot=False):
    X_bias = add_bias_column(X)
    y = y.flatten()  
    theta0 = np.zeros(X_bias.shape[1])

    result = minimize(
        fun=cost_function,
        x0=theta0,
        args=(X_bias, y),
        method='BFGS'
    )

    theta_opt = result.x
    y_pred = X_bias @ theta_opt
    mse = np.mean((y - y_pred) ** 2)

    print("=" * 40)
    print(f"θ: {theta_opt}")
    print(f"MSE: {mse:.4f}")

    if plot and X.shape[1] == 1:
        plt.figure()
        plt.scatter(X[:, 0], y, color='red', label='Dados reais')
        plt.plot(X[:, 0], y_pred, label='Ajuste')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Regressão Linear Univariada')
        plt.legend()
        plt.show()

    return theta_opt, y_pred, mse


# ================================
#              Testes 
# ================================
if __name__ == "__main__":
    # Univariado
    print("\nRegressão Univariada:")
    x_uni = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_uni = np.array([2.0, 2.8, 3.6, 4.5])
    regressao_otimizacao(x_uni, y_uni, plot=True)

    # Multivariado
    print("\nRegressão Multivariada:")
    X_multi = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0]
    ])
    y_multi = np.array([2.1, 2.5, 4.9, 5.1])
    regressao_otimizacao(X_multi, y_multi, plot=False)
