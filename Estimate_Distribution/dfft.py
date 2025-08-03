import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def unknown_function(x):
    return np.exp(x)
def rmse(list1, list2):
    
    list1 = np.array(list1)
    list2 = np.array(list2)
    mse = np.mean((list1 - list2) ** 2)
    rmse_value = np.sqrt(mse)
    return rmse_value
def compute_fourier(array_of_points):
    x = np.array(array_of_points)
    #I thought it was cheating to place all 
    #start_index = len(x) // 2 - 50
    #end_index = start_index + 100
    #x = x[start_index:end_index]
    #print(len(x))
    X = np.fft.fft(x)

    x_reconstructed = np.fft.ifft(X)
    return x,x_reconstructed.real


#def read_csv():
#    return pd.read_csv(r'C:\Users\evgisar\OneDrive - Ericsson\Desktop\Princ. Max Entr\data\data.csv').copy()
#df = read_csv()
#
#for col in df.columns:
#    if df[col].dtype == 'object':
#        continue
#    else:
#       x,y = compute_fourier(df[col])
#        print(f'RMSE of {col}: {rmse(x,y)}')
#



x_values = [x for x in range(-100,100)]
y_values = [unknown_function(x) for x in x_values]


original, reconstructed = compute_fourier(y_values)

#Cant comput rmse anymore because its unequal arrays

#error = rmse(y_values, reconstructed)
#print(error)

plt.figure(figsize=(12, 6))
plt.plot(range(len(original)), original, label='Original Function')
plt.plot(range(len(reconstructed)), reconstructed, label='Reconstructed Function', linestyle='dashed')
plt.title('Original and Reconstructed Functions')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('dfft')
plt.show()
print(rmse(original,reconstructed))