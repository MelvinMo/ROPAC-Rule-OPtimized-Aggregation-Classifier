from scipy.io import arff
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from Class import *


"""
# This is the code for ROPAC-L; to change it to ROPAC-M, you should uncomment a specific line in the Class file. For more information, read the comments there.

# Convert the ARFF data to a structured NumPy array
data_array = np.array(data.tolist(), dtype=data.dtype)
print("Data Array Shape:", data_array.shape)

# Separate features (X) and labels (y)
X = data_array[:,:-1] # Features
y = data_array[:,-1]  # Labels

# Modify the dataset path below to test other ARFF files.
"""
data, meta = arff.loadarff("/data/28-haberman.arff")
dataTypes = meta.types();
dataSet = pd.DataFrame(data).values;
preprocessor = ROPACPreprocesser();
X, y = preprocessor.fit_transform(dataSet, dataTypes);                 
accuracy = 0 ;
numOfRules = 0;
n_splits1 = 10;
kf = KFold(n_splits=n_splits1 , random_state=1 , shuffle=True)
i=0
for train_index, test_index in kf.split(X):
    X_train , X_test , Y_train , Y_test = [] , [] , [] , []
    X_train , X_test = X[train_index] , X[test_index]
    Y_train , Y_test = y[train_index] , y[test_index]
    ropac = None
    ropac = ROPAC(alpha = 0.99, gamma = 0.6, suppress_warnings = True);
    ropac.fit(X_train, Y_train);
    ropac.reduceRules();
    score = ropac.score(X_test, Y_test);
    rules = ropac.getNumOfRules();
    numOfRules += rules;
    accuracy += score;
    print("Results for the fold {0}:".format(i));
    print("Accuracy: {0}".format(score));
    print("Number of rules generated: {0}".format(rules));
    print("\n");
    i=i+1


print("#####################\n");
print("\tFinal results for {0} folds:".format(n_splits1))
print("\tTotal number of rules generated: {0}".format(numOfRules/10));
print("\tFinal accuracy in percentage: {:.2f}".format((accuracy/10)*100));    
print("\n#####################");