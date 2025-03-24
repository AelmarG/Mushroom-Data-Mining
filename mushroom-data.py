#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import the data and label the columns as they are unlabelled
mushrooms = pd.read_csv('./agaricus-lepiota.data',
                        names=['classification','cap-shape','cap-surface','cap-color',
                               'bruises?','odor','gill-attachment',
                               'gill-spacing','gill-size','gill-color',
                               'stalk-shape','stalk-root','stalk-surface-above-ring',
                               'stalk-surface-below-ring','stalk-color-above-ring',
                               'stalk-color-below-ring','veil-type',
                               'veil-color','ring-number','ring-type',
                               'spore-print-color','population','habitat'])

###PRE-PROCESSING

#turn any '?' into a NaN
mushrooms = mushrooms.replace(to_replace='?', value=np.nan)

#drop every row that is missing data (rows with NaN)
mushrooms = mushrooms.dropna()


###CLASSIFICATION METHOD 1: DECISION TREES

#split the attributes of the data frame into the class value and all other attributes
X = mushrooms.drop(['classification'], axis=1)
Y = mushrooms['classification']

#split X and Y into testing and training sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.2,        #testing is 20% of the data
    stratify=Y,           #preserve class distribution
    random_state=42       #ensures same split every run
)

###CLASSIFICATION METHOD 2: NAIVE BAYES



####CLASSIFICATOIN METHOD 3: SUPPORT VECTOR MACHINES
