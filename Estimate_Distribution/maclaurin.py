import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import random
def unknown_function(x):

    return math.sin(x)
    

def derivate_1(data, i, h):
    if i > 0 and i < len(data) - 1:
        return (data[i+1] - data[i-1]) / (2*h)
    else:
        return None

def derivate_2(data, i, h):
    if i > 0 and i < len(data) - 1:
        return (data[i+1] - 2*data[i] + data[i-1]) / (h**2)
    else:
        return None

def derivate_3(data, i, h):
    if i > 1 and i < len(data) - 2:
        return (data[i+2] - 2*data[i+1] + 2*data[i-1] - data[i-2]) / (2*(h**3))
    else:
        return None

def derivate_4(data, i, h):
    if i > 1 and i < len(data) - 2:
        return (-data[i+2] + 4*data[i+1] - 6*data[i] + 4*data[i-1] - data[i-2]) / (h**4)
    else:
        return None
def actual_taylor_e(x):
    return 1 + x + x**2/math.factorial(2)+x**3/math.factorial(3)
def actual_taylor_ln(x):
    return (x-1) - ((x-1)**2)/2 + ((x-1)**3)/3 - ((x-1)**4)/4

def taylor_series(data, x, h=1):
    center_index = len(data)//2
    d1 = derivate_1(data, center_index, h)
    d2 = derivate_2(data, center_index, h)
    d3 = derivate_3(data, center_index, h)
    d4 = derivate_4(data, center_index, h)
    
    terms = []
    if d1 is not None:
        terms.append((d1 * (x**1)) / math.factorial(1))
    if d2 is not None:
        terms.append((d2 * (x**2)) / math.factorial(2))
    if d3 is not None:
        terms.append((d3 * (x**3)) / math.factorial(3))
    if d4 is not None:
        terms.append((d4 * (x**4)) / math.factorial(4))
    
    return sum(terms)
start = -80
end = 100
sample_points = [unknown_function(x) for x in range(start, end)]



xpoints = np.arange(start, end)
ypoints = np.array(sample_points)

plt.plot(xpoints, ypoints, label="Original Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Original Function vs Taylor Series Approximation Plot")
plt.legend()

taylor_approximations = [taylor_series(sample_points, x) for x in range(start, end)]
#taylor_approximations = [actual_taylor_ln(x) for x in range(start,end)]
plt.plot(xpoints, taylor_approximations, label="Taylor Series Approximation")
plt.legend()
plt.savefig('polynomial.png')
plt.show()
sample_points = np.array(sample_points)
taylor_approximations = np.array(taylor_approximations)


print(f'RMSE: {sqrt(mean_squared_error(sample_points, taylor_approximations))}')

