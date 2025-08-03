import sympy as smp


lambda_0, lambda_1, x, u = smp.symbols('lambda_0 lambda_1 x u', real=True)
a, b = smp.symbols('a b', real=True)

'''Define your function'''
lagrange = smp.exp(1 - lambda_1)

'''Integrate our function 'lagrange' with respect to x with the bounds [a,b]'''
integral = smp.integrate(lagrange, (x, a, b))

'''We Need to setup the equation to solve so in our case it is the integral = 1. Indicating That the sum of the probabilites = 1'''
solution = smp.solve(integral - 1, lambda_1)

solved_function = smp.exp(1 - solution[0])


print(f'Integral {integral}\n')
print(f'Solution {solution}\n')
print(f'Solved Function {solved_function}\n')

