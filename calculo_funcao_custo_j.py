import numpy as np

def total_cost(w, b, x, y):
    m = len(x)
    J = 0
    for i in range(m):
        J += (1 - y[i])*(w*x[i] + b) + np.log(1 + np.exp(-(w*x[i] + b)))
    J_total = J / m
    return J_total

x = np.array([2, 4, 6, 8])
y = np.array([0, 0, 1, 1])

print("Variações em w e b:")
print("J(w=0.5, b=-5.5) =", total_cost(0.5, -5.5, x, y))
print("J(w=0.5, b=-6)  =", total_cost(0.5, -6, x, y))
print("J(w=0.5, b=-6.5)=", total_cost(0.5, -6.5, x, y))
print("J(w=1, b=-5.5) =", total_cost(1, -5.5, x, y))
print("J(w=1, b=-6)  =", total_cost(1, -6, x, y))
print("J(w=1, b=-6.5)=", total_cost(1, -6.5, x, y))
print("J(w=1.5, b=-5.5) =", total_cost(1.5, -5.5, x, y))
print("J(w=1.5, b=-6)  =", total_cost(1.5, -6, x, y))
print("J(w=1.5, b=-6.5)=", total_cost(1.5, -6.5, x, y))
print("===========================================")
print("Pequenas variações em b:")
print("J(w=1, b=-6) =", total_cost(1, -6, x, y))
print("J(w=1, b=-5.9) =", total_cost(1, -5.9, x, y))
print("J(w=1, b=-5.8) =", total_cost(1, -5.8, x, y))
print("J(w=1, b=-5.7) =", total_cost(1, -5.7, x, y))
print("J(w=1, b=-5.6) =", total_cost(1, -5.6, x, y))
print("J(w=1, b=-5.5) =", total_cost(1, -5.5, x, y))
print("J(w=1, b=-5.4) =", total_cost(1, -5.4, x, y))
print("J(w=1, b=-5.3) =", total_cost(1, -5.3, x, y))
print("J(w=1, b=-5.2) =", total_cost(1, -5.2, x, y))
print("J(w=1, b=-5.1) =", total_cost(1, -5.1, x, y))
print("J(w=1, b=-5.0) =", total_cost(1, -5.0, x, y))
print("===========================================")
print("Pequenas variações em w:")
print("J(w=0.5, b=-5) =", total_cost(0.5, -5, x, y))
print("J(w=0.6, b=-5) =", total_cost(0.6, -5, x, y))
print("J(w=0.7, b=-5) =", total_cost(0.7, -5, x, y))
print("J(w=0.8, b=-5) =", total_cost(0.8, -5, x, y))
print("J(w=0.9, b=-5) =", total_cost(0.9, -5, x, y))
print("J(w=1, b=-5) =", total_cost(1, -5, x, y))
print("J(w=1.1, b=-5) =", total_cost(1.1, -5, x, y))
print("J(w=1.2, b=-5) =", total_cost(1.2, -5, x, y))
print("J(w=1.3, b=-5) =", total_cost(1.3, -5, x, y))
print("J(w=1.4, b=-5) =", total_cost(1.4, -5, x, y))
print("J(w=1.5, b=-5) =", total_cost(1.5, -5, x, y))
