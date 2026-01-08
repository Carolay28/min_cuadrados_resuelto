import numpy as np
import matplotlib.pyplot as plt
import math
from src import ajustar_min_cuadrados


def der_parcial_q2(xs, ys):
    c2 = 0
    c1 = 0
    c0 = 0
    c_ind = 0
    for xi, yi in zip(xs, ys):
        c2 += xi**4
        c1 += xi**3
        c0 += xi**2
        c_ind += yi * xi**2
    return (c2, c1, c0, c_ind)


def der_parcial_q1(xs, ys):
    c2 = 0
    c1 = 0
    c0 = 0
    c_ind = 0
    for xi, yi in zip(xs, ys):
        c2 += xi**3
        c1 += xi**2
        c0 += xi
        c_ind += yi * xi
    return (c2, c1, c0, c_ind)


def der_parcial_q0(xs, ys):
    c2 = 0
    c1 = 0
    c0 = 0
    c_ind = 0
    for xi, yi in zip(xs, ys):
        c2 += xi**2
        c1 += xi
        c0 += 1
        c_ind += yi
    return (c2, c1, c0, c_ind)


xs1 = [
    -5.0000, -3.8889, -2.7778, -1.6667, -0.5556,
    0.5556, 1.6667, 2.7778, 3.8889, 5.0000
]

ys1 = [
    57.2441, 33.0303, 16.4817, 7.0299, 0.5498,
    0.7117, 3.4185, 12.1767, 24.9167, 44.2495
]


a2, a1, a0 = ajustar_min_cuadrados(xs1, ys1, gradiente=[
    der_parcial_q2,
    der_parcial_q1,
    der_parcial_q0
])

x = np.linspace(-5, 5, 200)
y = [a2 * xi**2 + a1 * xi + a0 for xi in x]

plt.scatter(xs1, ys1)
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Conjunto de datos 1 - Ajuste cuadrático")
plt.show()


def der_parcial_e1(xs, ys):
    c1 = 0
    c0 = 0
    c_ind = 0
    for xi, yi in zip(xs, ys):
        ex = math.exp(xi)
        c1 += ex * ex
        c0 += ex
        c_ind += yi * ex
    return (c1, c0, c_ind)


def der_parcial_e0(xs, ys):
    c1 = 0
    c0 = 0
    c_ind = 0
    for xi, yi in zip(xs, ys):
        ex = math.exp(xi)
        c1 += ex
        c0 += 1
        c_ind += yi
    return (c1, c0, c_ind)


xs2 = [
    0.0003, 0.0822, 0.2770, 0.4212, 0.4403,
    0.5588, 0.5943, 0.6134, 0.9070, 1.0367,
    1.1903, 1.2511, 1.2519, 1.2576, 1.6165,
    1.6761, 2.0114, 2.0557, 2.1610, 2.6344
]

ys2 = [
    1.1017, 1.5021, 0.3844, 1.3251, 1.7206,
    1.9453, 0.3894, 0.3328, 1.2887, 3.1239,
    2.1778, 3.1078, 4.1856, 3.3640, 6.0330,
    5.8088, 10.5890, 11.5865, 11.8221, 26.5077
]


a, b = ajustar_min_cuadrados(xs2, ys2, gradiente=[
    der_parcial_e1,
    der_parcial_e0
])

x2 = np.linspace(min(xs2), max(xs2), 200)
y2 = [a * math.exp(b * xi) for xi in x2]

plt.scatter(xs2, ys2)
plt.plot(x2, y2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Conjunto de datos 2 - Ajuste exponencial")
plt.show()
