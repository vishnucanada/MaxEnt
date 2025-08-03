
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import configparser
import seaborn as sns
from scipy import stats

def is_close(a,b):
    return abs(a-b) < 5

def assume_dataset(data):
    mean = np.mean(data)
    median = np.median(data)
   
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    '''We should set a Threshold for similarity'''

    print(mean,median,skewness,kurtosis)

    if is_close(mean,median) and (is_close(skewness,0) and is_close(kurtosis,0)):
        print('Normal Dist')
        test_prediction(data,'normal')
    if is_close(mean,median) and is_close(skewness,0) and kurtosis <= 1:
        print('Uniform Dist')
        test_prediction(data,'uniform')
    if (not is_close(mean,median)) and skewness>0 and kurtosis>1:
        print('Expo Dist')
        test_prediction(data,'exponential')
    
    
def test_prediction(actual_data,distribution):


    mean = np.mean(actual_data)
    std = np.std(actual_data)
    data_min = np.min(actual_data)
    data_max = np.max(actual_data)
    scale = np.mean(actual_data)

    if distribution == 'normal':
        sample_distribution = np.random.normal(loc=mean, scale=std, size=len(actual_data))
    elif distribution == 'uniform':
        sample_distribution = np.random.uniform(low=data_min, high=data_max, size=len(actual_data))
    elif distribution == 'exponential':
        sample_distribution = np.random.exponential(scale=scale, size=len(actual_data))


    _, ax = plt.subplots()
    
    sns.kdeplot(actual_data, label='Actual Data', ax=ax)
    sns.kdeplot(sample_distribution,label=distribution,ax=ax)
    """sns.kdeplot(sample_normal, label='Normal', ax=ax)
    sns.kdeplot(sample_uniform, label='Uniform', ax=ax)
    sns.kdeplot(sample_exponential, label='Exponential', ax=ax)"""
    
    ax.legend()
    ax.set_title('Density Plot')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    plt.show()

def read_dataset():
    return pd.read_csv(r'C:\Users\evgisar\OneDrive - Ericsson\Desktop\Princ. Max Entr\data\data.csv')


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    #data = config['PATHS']['data_path']
    df = read_dataset()

    for col in df.columns:
        if isinstance(df[col][0],str):
            continue
        else:
            assume_dataset(df[col].dropna())

main()