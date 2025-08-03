import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint
from scipy.integrate import quad
def pdf(x, lambdas, average):
    exponent = np.sum([l * (x - average) ** i for i, l in enumerate(lambdas)], axis=0)
    return np.exp(np.clip(exponent, -700, 700))  # Clip to avoid overflow

def log_likelihood(lambdas, data):
    average = np.mean(data)
    if np.any(np.abs(lambdas) > 1e5):
        return np.inf
    
    pdf_values = pdf(data, lambdas, average)
    log_pdf_values = np.log(pdf_values + 1e-10)  # Add small constant to avoid log(0)
    return -np.sum(log_pdf_values)

def normalization_constraint(lambdas, data):
    average = np.mean(data)
    integral, _ = quad(lambda x: pdf(x, lambdas, average), np.min(data), np.max(data))
    return integral - 1

def moment_constraint(lambdas, data, moment_order):
    average = np.mean(data)
    moment_empirical = np.mean((data - average) ** moment_order)
    integral, _ = quad(lambda x: (x - average) ** moment_order * pdf(x, lambdas, average), np.min(data), np.max(data))
    return integral - moment_empirical

def maximize_log_likelihood(initial_guess, data):
    constraints = [NonlinearConstraint(lambda lambdas: normalization_constraint(lambdas, data), -0.1, 0.1)]
    for moment_order in range(1, len(initial_guess)):
        constraints.append(NonlinearConstraint(lambda lambdas: moment_constraint(lambdas, data, moment_order), -0.1, 0.1))

    bounds = [(-1e5, 1e5)] * len(initial_guess)
    result = minimize(log_likelihood, initial_guess, args=(data,), method='trust-constr', 
                      constraints=constraints, bounds=bounds, options={'verbose': 1, 'maxiter': 5000})
    return result.x

def quantum_fidelity(p, q):
    return np.sum(np.sqrt(p * q))


def negative_quantum_fidelity(lambdas, data, bins):
    average = np.mean(data)
    
    hist, _ = np.histogram(data, bins=bins, density=True)
    
    x = (bins[:-1] + bins[1:]) / 2  # bin centers
    predicted = pdf(x, lambdas, average)
    predicted /= np.sum(predicted)  # normalize
    
    return -quantum_fidelity(hist, predicted)

def maximize_fidelity(initial_guess, data, bins):
    constraints = [NonlinearConstraint(lambda lambdas: normalization_constraint(lambdas, data), -0.1, 0.1)]
    for moment_order in range(1, len(initial_guess)):
        constraints.append(NonlinearConstraint(lambda lambdas: moment_constraint(lambdas, data, moment_order), -0.1, 0.1))

    bounds = [(-1e5, 1e5)] * len(initial_guess)
    result = minimize(negative_quantum_fidelity, initial_guess, args=(data, bins), method='trust-constr', 
                      constraints=constraints, bounds=bounds, options={'verbose': 1, 'maxiter': 1000})
    return result.x

def plot_distributions(data, bins, lambdas_mle, lambdas_fidelity, col_name):
    average = np.mean(data)
    x = np.linspace(data.min(), data.max(), 1000)
    
    pdf_mle = pdf(x, lambdas_mle, average)
    pdf_fidelity = pdf(x, lambdas_fidelity, average)
    
    plt.figure(figsize=(12, 6))
    plt.hist(data, bins=bins, density=True, alpha=0.5, label='Empirical Distribution')
    plt.plot(x, pdf_mle, label='MLE Predicted PDF', color='red')
    plt.plot(x, pdf_fidelity, label='Fidelity Predicted PDF', color='green')
    plt.title(f'Distribution for {col_name}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.savefig(f'cars_data\\{col_name}.png')
    #plt.show()

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
if __name__ == '__main__':
    #path = r'C:\Users\evgisar\OneDrive - Ericsson\Desktop\Princ. Max Entr\data\wind_turbine_data.csv'
    path =  r'C:\Users\evgisar\OneDrive - Ericsson\Desktop\Princ. Max Entr\data\cars_data.csv'
    df = pd.read_csv(path).copy()

    initial_guess = [0.0, 0.0, 0.0, 0.0] 
    for col in df.columns:
        if df[col].dtype == 'object':
            continue
        else:
            # you dont need to always normalize data
            data = normalize_data(df[col].head(2500).dropna()).values
            
        
            try:
                
                optimal_lambdas_mle = maximize_log_likelihood(initial_guess, data)
                print(f"Optimal lambdas (MLE) for {col} with initial guess {initial_guess}:", optimal_lambdas_mle)
                
            
                bins = np.linspace(data.min(), data.max(), 50)  # Adjust number of bins as needed
                optimal_lambdas_fidelity = maximize_fidelity(initial_guess, data, bins)
                print(f"Optimal lambdas (Fidelity) for {col} with initial guess {initial_guess}:", optimal_lambdas_fidelity)
                
                # Plot the distributions
                plot_distributions(data, bins, optimal_lambdas_mle, optimal_lambdas_fidelity, col)
            except Exception as e:
                print(f"Error processing column {col} with initial guess {initial_guess}: {str(e)}")
            
            print("\n")
