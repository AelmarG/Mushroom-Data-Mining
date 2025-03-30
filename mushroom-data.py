#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


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

#create a copy to preserve original if needed
mushrooms_encoded = mushrooms.copy()

features_to_encode = mushrooms.columns.drop('classification')
mushrooms_encoded = pd.get_dummies(mushrooms, columns=features_to_encode, drop_first=True)

#show the first few rows
print(mushrooms_encoded.head())

###CLASSIFICATION METHOD 1: DECISION TREES

#split the attributes of the data frame into the class value and all other attributes
X = mushrooms_encoded.drop(['classification'], axis=1)
Y = mushrooms_encoded['classification']

#split X and Y into testing and training sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.01,        #testing is 40% of the data
    stratify=Y,           #preserve class distribution
    random_state=42       #ensures same split every run
)


######might have to check the decision tree stuff cuz the accuracies are scary close

#create the decision tree
dec_tree = tree.DecisionTreeClassifier(max_depth=4, random_state=42)

#train the model
dec_tree.fit(X_train, Y_train)

#predict on training and test sets
Y_train_pred = dec_tree.predict(X_train)
Y_test_pred = dec_tree.predict(X_test)

#compute accuracies
train_acc = accuracy_score(Y_train, Y_train_pred)
test_acc = accuracy_score(Y_test, Y_test_pred)

print(f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

plt.figure(figsize=(20, 10))
plot_tree(dec_tree, 
          feature_names=X.columns, 
          class_names=['edible', 'poisonous'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Decision Tree for Mushroom Classification")
plt.show()

###CLASSIFICATION METHOD 2: NAIVE BAYES



####CLASSIFICATOIN METHOD 3: SUPPORT VECTOR MACHINES
