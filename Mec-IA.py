import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

file = pd.read_csv("Automobile.csv")

# print(file.head())

# show graphics about the csv file
# for label in file.columns[:-1]:
#     plt.hist(file[file["origin"]=="usa"][label], color ='blue', label='gamma', alpha = 0.7, density = True)
#     plt.hist(file[file["origin"]=="europe"  ][label], color ='green', label='gamma', alpha = 0.7, density = True)
#     plt.hist(file[file["origin"]=="japan"][label], color ='red', label='gamma', alpha = 0.7, density = True)
#     plt.xlabel(label)
#     plt.ylabel(label)
#     plt.legend()
#     plt.show()
    
train, valid, test = np.split(file.sample(frac=1), [int(0.6*len(file)), int(0.8*len(file))])

print(len(train[train["origin"] == "usa"]))
print(len(train[train["origin"] == "japan"]))
print(len(train[train["origin"] == "europe"]))


def scale_dataset(file, oversample=False):
    X = file[file.columns[:-1]].values  
    y = file[file.columns[-1]].values   

    scaler = StandardScaler()
    X = scaler.fit_transform(X)  

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y