import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')


def generate_basis(x, n, type='2'):
    basis = []
    '''
    Testing new basis functions

    '''
    
    if type == '1':
        return np.exp(-np.outer(x, np.arange(n+1)))
    elif type == '2':
        return np.power.outer(x, np.arange(n+1))
    elif type == '3':
        basis = np.ones((len(x), 2*(n+1)))
        basis[:, 1] = x
        for i in range(n):
            basis[:, 2*i + 2] = np.sin(2 * np.pi * (i+1) ** x)
            basis[:, 2*i + 3] = np.cos(2 * np.pi * (i+1) ** x)
        return basis

def compute_fidelity(numbers1, numbers2, bin_percentage=0.05):
    n1 = len(numbers1)
    n2 = len(numbers2)
    
    # Determine number of bins based on the larger dataset
    num_bins = math.ceil(bin_percentage * max(n1, n2))
    
    # Calculate common bin edges
    min_val = min(min(numbers1), min(numbers2))
    max_val = max(max(numbers1), max(numbers2))
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    def convert_to_occurrence_probabilities(numbers, bin_edges):
        # Digitize data into bins
        bin_indices = np.digitize(numbers, bins=bin_edges, right=True)
        bin_indices -= 1  # Adjust to zero-indexed bins

        # Calculate probabilities for each bin
        total_elements = len(bin_indices)
        counts = Counter(bin_indices)
        bin_probabilities = {bin_idx: count / total_elements for bin_idx, count in counts.items()}

        bin_range_probabilities = {
            f'[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]': bin_probabilities.get(i, 0)
            for i in range(len(bin_edges) - 1)
        }

        return bin_range_probabilities
    
    probabilities1 = convert_to_occurrence_probabilities(numbers1, bin_edges)
    probabilities2 = convert_to_occurrence_probabilities(numbers2, bin_edges)
    
    return fidelity(probabilities1,probabilities2)

def convert_to_occurrence_probabilities(numbers, bin_percentage=0.05):
    n = len(numbers)
    
    # Determine number of bins based on percentage of data points
    num_bins = math.ceil(bin_percentage * n)
    
    # Calculate bin edges
    min_val = min(numbers)
    max_val = max(numbers)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    # Digitize data into bins
    bin_indices = np.digitize(numbers, bins=bin_edges, right=True)
    bin_indices -= 1  # Adjust to zero-indexed bins
    
    # Calculate probabilities for each bin
    total_elements = len(bin_indices)
    counts = Counter(bin_indices)
    bin_probabilities = {bin_idx: count / total_elements for bin_idx, count in counts.items()}
    
    bin_range_probabilities = {
        f'[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]': bin_probabilities.get(i, 0)
        for i in range(num_bins)
    }
    
    return bin_range_probabilities



def calculate_entropy(probabilities):
    entropy = 0
    
    for prob in probabilities:
        if prob > 0:
            entropy = entropy + prob * math.log2(prob)
        else:
            continue
    return -1*entropy
def plot(x, y, predictions, rmse, r2, name, entropy_actual, entropy_predicted,fidelity,pdf_data1=None, pdf_data2=None):
    plt.style.use('ggplot')
    _, axs = plt.subplots(3, 1, figsize=(8, 9))

    axs[0].plot(x, y,color='#0074D9') 
    axs[0].plot(x, predictions, linestyle='dashed',color='#FF851B')  

    axs[0].legend()

    axs[0].text(0.05, 0.95, f'RMSE: {rmse:.2f}', transform=axs[0].transAxes, verticalalignment='top')
    axs[0].text(0.05, 0.90, f'R2: {r2:.2f}', transform=axs[0].transAxes, verticalalignment='top')

    axs[0].set_title('Original Data vs Reconstructed Function')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Value')
    axs[0].grid(True)

    plot_pdf_from_dict(pdf_data1, ax=axs[1])

    axs[1].legend()
    axs[1].text(0.95, 0.95, f'Entropy Actual: {entropy_actual:.5f}', transform=axs[1].transAxes, verticalalignment='center_baseline', horizontalalignment='right')
    axs[1].text(0.95, 0.87, f'Entropy Simulated: {entropy_predicted:.5f}', transform=axs[1].transAxes, verticalalignment='center_baseline', horizontalalignment='right')
    axs[1].text(0.95, 0.79, f'Fidelity: {fidelity:.5f}', transform=axs[1].transAxes, verticalalignment='center_baseline', horizontalalignment='right')
    #axs[1].text(0.95, 0.71, f'Gaussian Entropy: {0.5 * np.log2(2 * np.pi * np.e * np.std(y)**2):.5f}', transform=axs[1].transAxes, verticalalignment='center_baseline', horizontalalignment='right')


    axs[1].set_title('Probability Density Function 1 (PDF)')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('Density')
    axs[1].grid(True)

    
    plot_pdf_from_dict(pdf_data2, ax=axs[2], color='#FF851B')
    axs[2].legend()

    axs[2].set_title('Probability Density Function 2 (PDF)')
    axs[2].set_xlabel('Value')
    axs[2].set_ylabel('Density')
    axs[2].grid(True)

    plt.tight_layout()
    #plt.savefig('data.png')
    plt.show()

def plot_pdf_from_dict(probabilities, ax=None, color='#0074D9'):
    if ax is None:
        ax = plt.gca()

    num_intervals = len(probabilities)
    interval_width = 1  # Adjust as necessary

    x_values = np.linspace(0, num_intervals * interval_width, num_intervals, endpoint=False)

    ax.bar(x_values, probabilities.values(), width=interval_width, align='center', alpha=0.7, color=color)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.grid(True)
def fidelity(phi, psi):
    keys = phi.keys() & psi.keys()
    fidelity = 0
    
    for key in keys:
        fidelity += np.sqrt(phi[key] * psi[key])
    
    return fidelity

def analysis(data):
 
    x = np.arange(len(data))
    y = data
    n = len(data) # In our expirements increasing the sample points for a quantum based basis function increases the certainty
    for i in ['1','2','3']:
        try:

            design_matrix = np.vstack(generate_basis(x, n, i))
            
            coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)

            predictions = np.dot(design_matrix, coefficients)
            

            rmse = np.sqrt(root_mean_squared_error(predictions, y))
            r2 = r2_score(predictions, y)


            if i == '1':
                name = 'Exponential'
            elif i == '2':
                name = 'Polynomial'
            elif i == '3':
                name = 'Trigometric'

            actual_data = convert_to_occurrence_probabilities(y)
            simulated_data = convert_to_occurrence_probabilities(predictions)

            entropy = calculate_entropy(actual_data.values())
            entropy_1 = calculate_entropy(simulated_data.values())
            fidelity = compute_fidelity(predictions,y)
            plot(x,y,predictions,rmse,r2,name,entropy,entropy_1,fidelity,actual_data,simulated_data)
            

        except np.linalg.LinAlgError:
            print('Did not converge')
            continue
        except RuntimeError:
            continue
