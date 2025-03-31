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
X_norm = mushrooms_encoded.drop(['classification'], axis=1)
Y_norm = mushrooms_encoded['classification']

#split X and Y into testing and training sets
X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = train_test_split(
    X_norm, Y_norm, 
    test_size=0.4,        #testing is 40% of the data
    stratify=Y_norm,           #preserve class distribution
    random_state=42       #ensures same split every run
)

#create the decision tree
norm_dec_tree = tree.DecisionTreeClassifier(max_depth=4, random_state=42)

#train the model
norm_dec_tree.fit(X_train_norm, Y_train_norm)

#predict on training and test sets
Y_train_pred_norm = norm_dec_tree.predict(X_train_norm)
Y_test_pred_norm = norm_dec_tree.predict(X_test_norm)

#compute accuracies
norm_tree_train_acc = accuracy_score(Y_train_norm, Y_train_pred_norm)
norm_tree_test_acc = accuracy_score(Y_test_norm, Y_test_pred_norm)

print(f"Tree Train Acc W/O Feature Selection = {norm_tree_train_acc:.4f}") 
print(f"Tree Test Acc W/O Feature Selection = {norm_tree_test_acc:.4f}", '\n')

plt.figure(figsize=(20, 10))
plot_tree(norm_dec_tree, 
          feature_names=X_norm.columns, 
          class_names=['edible', 'poisonous'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Decision Tree for Mushroom Classification")
plt.show()

###CLASSIFICATION METHOD 1.5: DECISION TREES WITH FEATURE SELECTION

###########this also might need to be checked later I got the exact same accuracies :|

#create the RFE for feature selection
rfe = RFE(norm_dec_tree, n_features_to_select=5)
selected = rfe.fit_transform(X_train_norm, Y_train_norm)
print("Selected features:", X_train_norm.columns[rfe.get_support()])

#create a dataframe using only the selected features
mushrooms_fs = pd.DataFrame()

for i in X_train_norm.columns[rfe.get_support()]:
    mushrooms_fs[i] = mushrooms_encoded[i]

#create a new decision tree for the selected features
dec_tree_fs = tree.DecisionTreeClassifier(max_depth=4, random_state=42)

#split the attributes again
X_fs = mushrooms_fs
Y_fs = Y_norm

#split the data into test and train sets
X_train_fs, X_test_fs, Y_train_fs, Y_test_fs = train_test_split(
    X_fs, Y_fs,
    test_size = 0.4,
    stratify = Y_fs,
    random_state = 42
)

#fit the decision tree
dec_tree_fs.fit(X_train_fs, Y_train_fs)

#predict on training and test sets
Y_train_pred_fs = dec_tree_fs.predict(X_train_fs)
Y_test_pred_fs = dec_tree_fs.predict(X_test_fs)

#compute accuracies
fs_train_acc = accuracy_score(Y_train_fs, Y_train_pred_fs)
fs_test_acc = accuracy_score(Y_test_fs, Y_test_pred_fs)

print(f"Train Acc With Feature Selection = {fs_train_acc:.4f}") 
print(f"Test Acc With Feature Selection = {fs_test_acc:.4f}")

###CLASSIFICATION METHOD 2: NAIVE BAYES



####CLASSIFICATOIN METHOD 3: SUPPORT VECTOR MACHINES
