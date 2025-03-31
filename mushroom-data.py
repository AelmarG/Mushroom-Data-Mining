#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


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
    test_size=0.2,        #testing is 40% of the data
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
plt.title("Decision Tree for Mushroom Classification Without Feature Selection")
plt.show()

###CLASSIFICATION METHOD 1.5: DECISION TREES WITH FEATURE SELECTION

#create the RFE for feature selection
rfe = RFE(norm_dec_tree, n_features_to_select=30)
selected = rfe.fit_transform(X_train_norm, Y_train_norm)
print("Selected features:", X_train_norm.columns[rfe.get_support()], '\n')

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
fs_tree_train_acc = accuracy_score(Y_train_fs, Y_train_pred_fs)
fs_tree_test_acc = accuracy_score(Y_test_fs, Y_test_pred_fs)

print(f"Tree Train Acc With Feature Selection = {fs_tree_train_acc:.4f}") 
print(f"Tree Test Acc With Feature Selection = {fs_tree_test_acc:.4f}", '\n')

plt.figure(figsize=(20, 10))
plot_tree(dec_tree_fs, 
          feature_names=X_fs.columns, 
          class_names=['edible', 'poisonous'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Decision Tree for Mushroom Classification With Feature Selection")
plt.show()

###CLASSIFICATION METHOD 2: NAIVE BAYES

#create the gaussian naive bayes for the normal dataframe
gb_norm = GaussianNB()

#train the model using the normal X and Y attributes
gb_norm.fit(X_train_norm, Y_train_norm)

#predict on test and training sets
Y_NB_train_pred_norm = gb_norm.predict(X_train_norm)
Y_NB_test_pred_norm = gb_norm.predict(X_test_norm)

norm_gb_train_acc = accuracy_score(Y_train_norm, Y_NB_train_pred_norm)
norm_gb_test_acc = accuracy_score(Y_test_norm, Y_NB_test_pred_norm)

print(f"NB Train Acc W/O Feature Selection = {norm_gb_train_acc:.4f}") 
print(f"NB Test Acc W/O Feature Selection = {norm_gb_test_acc:.4f}", '\n')

###CLASSIFICATION METHOD 2.5: NAIVE BAYES WITH FEATURE SELECTION

#create gaussian naive bayes for the feature selected dataframe
gb_fs = GaussianNB()

#train the model using the feature selected X and Y attributes
gb_fs.fit(X_train_fs, Y_train_fs)

#predict on test and training sets
Y_NB_train_pred_fs = gb_fs.predict(X_train_fs)
Y_NB_test_pred_fs = gb_fs.predict(X_test_fs)

fs_gb_train_acc = accuracy_score(Y_train_fs, Y_NB_train_pred_fs)
fs_gb_test_acc = accuracy_score(Y_test_fs, Y_NB_test_pred_fs)

print(f"NB Train Acc With Feature Selection = {fs_gb_train_acc:.4f}") 
print(f"NB Test Acc With Feature Selection = {fs_gb_test_acc:.4f}", '\n')

###CLASSIFICATOIN METHOD 3: SUPPORT VECTOR MACHINES

#create the SVM for the normal attributes
svm_norm = SVC()

#train the SVM
svm_norm.fit(X_train_norm, Y_train_norm)

#predict on test and training sets
Y_SVM_train_pred_norm = svm_norm.predict(X_train_norm)
Y_SVM_test_pred_norm = svm_norm.predict(X_test_norm)

norm_svm_train_acc = accuracy_score(Y_train_norm, Y_SVM_train_pred_norm)
norm_svm_test_acc = accuracy_score(Y_test_norm, Y_SVM_test_pred_norm)

print(f"SVM Train Acc W/O Feature Selection = {norm_svm_train_acc:.4f}") 
print(f"SVM Test Acc W/O Feature Selection = {norm_svm_test_acc:.4f}", '\n')

###CLASSIFICATOIN METHOD 3.5: SUPPORT VECTOR MACHINES WITH FEATURE SELECTION

#create the SVM for the feature selected attributes
svm_fs = SVC()

#train the SVM
svm_fs.fit(X_train_fs, Y_train_fs)

#predict on test and training sets
Y_SVM_train_pred_fs = svm_fs.predict(X_train_fs)
Y_SVM_test_pred_fs = svm_fs.predict(X_test_fs)

#calculate accuracies
fs_svm_train_acc = accuracy_score(Y_train_fs, Y_SVM_train_pred_fs)
fs_svm_test_acc = accuracy_score(Y_test_fs, Y_SVM_test_pred_fs)

print(f"SVM Train Acc With Feature Selection = {fs_svm_train_acc:.4f}") 
print(f"SVM Test Acc With Feature Selection = {fs_svm_test_acc:.4f}", '\n')

###CLASSIFICATION METHOD 4: K-NEAREST-NEIGHBOURS

#create the KNN for the normal attributes
knn_norm = KNeighborsClassifier()

#train the KNN
knn_norm.fit(X_train_norm, Y_train_norm)

#predict on the test and training sets
Y_KNN_train_pred_norm = knn_norm.predict(X_train_norm)
Y_KNN_test_pred_norm = knn_norm.predict(X_test_norm)

#calculate accuracies
norm_knn_train_acc = accuracy_score(Y_train_norm, Y_KNN_train_pred_norm)
norm_knn_test_acc = accuracy_score(Y_test_norm, Y_KNN_test_pred_norm)

print(f"KNN Train Acc W/O Feature Selection = {norm_knn_train_acc:.4f}") 
print(f"KNN Test Acc W/O Feature Selection = {norm_knn_test_acc:.4f}", '\n')

###CLASSIFICATION METHOD 4.5: K-NEAREST-NEIGHBOURS WITH FEATURE SELECTION

#create the KNN for the normal attributes
knn_fs = KNeighborsClassifier()

#train the KNN
knn_fs.fit(X_train_fs, Y_train_fs)

#predict on the test and training sets
Y_KNN_train_pred_fs = knn_fs.predict(X_train_fs)
Y_KNN_test_pred_fs = knn_fs.predict(X_test_fs)

#calculate accuracies
fs_knn_train_acc = accuracy_score(Y_train_fs, Y_KNN_train_pred_fs)
fs_knn_test_acc = accuracy_score(Y_test_fs, Y_KNN_test_pred_fs)

print(f"KNN Train Acc With Feature Selection = {fs_knn_train_acc:.4f}") 
print(f"KNN Test Acc With Feature Selection = {fs_knn_test_acc:.4f}", '\n')

###CLASSIFICATION METHOD 5: MULTI-LAYER PERCEPTRON

#create the MLP
mlp_norm = MLPClassifier()

#train the MLP on the normal attributes
mlp_norm.fit(X_train_norm, Y_train_norm)

#predict on the test and training sets
Y_MLP_train_pred_norm = mlp_norm.predict(X_train_norm)
Y_MLP_test_pred_norm = mlp_norm.predict(X_test_norm)

#calculate accuracies
norm_mlp_train_acc = accuracy_score(Y_train_norm, Y_MLP_train_pred_norm)
norm_mlp_test_acc = accuracy_score(Y_test_norm, Y_MLP_test_pred_norm)

print(f"MLP Train Acc W/O Feature Selection = {norm_mlp_train_acc:.4f}")
print(f"MLP Test Acc W/O Feature Selection = {norm_mlp_test_acc:.4f}", '\n')

###CLASSIFICATION METHOD 5.5: MULTI-LAYER PERCEPTRON WITH FEATURE SELECTION

#create the MLP
mlp_fs = MLPClassifier()

#train the MLP on the feature selected attributes
mlp_fs.fit(X_train_fs, Y_train_fs)

#predict on the test and training sets
Y_MLP_train_pred_fs = mlp_fs.predict(X_train_fs)
Y_MLP_test_pred_fs = mlp_fs.predict(X_test_fs)

#calculate accuracies
fs_mlp_train_acc = accuracy_score(Y_train_fs, Y_MLP_train_pred_fs)
fs_mlp_test_acc = accuracy_score(Y_test_fs, Y_MLP_test_pred_fs)

print(f"MLP Train Acc With Feature Selection = {fs_mlp_train_acc:.4f}")
print(f"MLP Test Acc With Feature Selection = {fs_mlp_test_acc:.4f}", '\n')
