import numpy as np

def dichotomous(f, a_i, b_i, epsilon, l):
    a = a_i
    b = b_i
    i = 0
    while (b - a) > l:
        print(f"{i:<6}{a:12.6f}{b:12.6f}{(b-a):12.6f}")
        lambda_k = (a + b)/2 - epsilon
        mu_k = (a + b)/2 + epsilon
        if f(lambda_k) < f(mu_k):
            b = mu_k
        else:
            a = lambda_k
        i += 1
    x_min = (a + b)/2
    f_min = f(x_min)
    return x_min, f_min

epsilon = 1e-5
l = 1e-3
f1 = lambda x: x**4 - 14*x**3 + 60*x**2 - 70*x
f2 = lambda x: (1/4)*x**4 - (5/3)*x**3 - 6*x**2 + 19*x - 7

x1, f1_min = dichotomous(f1, 0, 2, epsilon, l)
print(f"\nf1: x_min = {x1:.6f}, f(x_min) = {f1_min:.6f}\n\n")

x2, f2_min = dichotomous(f2, -4, 4, epsilon, l)
print(f"\nf2: x_min = {x2:.6f}, f(x_min) = {f2_min:.6f}")
