#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


#import the data and label the columns as they are unlabelled
adults = pd.read_csv('./adult.data',
                        names=['age','workclass','fnlwgt', 'education',
                               'education-num', 'marital-status', 'occupation',
                               'relationship', 'race', 'sex', 'capital-gain',
                               'capital-loss', 'hours-per-week', 'native-country','annual-income'])

###PRE-PROCESSING

#turn any '?' into a NaN
adults = adults.replace(to_replace='?', value=np.nan)

#drop every row that is missing data (rows with NaN)
adults = adults.dropna()

#create a copy to preserve original if needed
adults_encoded = adults.copy()

features_to_encode = ['workclass','education','marital-status',
                        'occupation','relationship','race','sex',
                        'native-country']
 
adults_encoded = pd.get_dummies(adults, columns=features_to_encode, drop_first=True)

#show the first few rows
print(adults_encoded.head())

###CLASSIFICATION METHOD 1: DECISION TREES

#split the attributes of the data frame into the class value and all other attributes
X_full = adults_encoded.drop(['annual-income'], axis=1)
Y_full = adults_encoded['annual-income']

#use train_test_split with train_size=10000 to obtain a representative sample.
X_sample, _, Y_sample, _ = train_test_split(X_full, Y_full,
                                            train_size=10000,
                                            stratify=Y_full,
                                            random_state=42)

#split X and Y into testing and training sets
X_train_norm, X_test_norm, Y_train_norm, Y_test_norm = train_test_split(
    X_sample, Y_sample, 
    test_size=0.4,        #testing is 40% of the data
    stratify=Y_sample,           #preserve class distribution
    random_state=42       #ensures same split every run
)

#Create SMOTE versions of the training data to counteract the oversampling in our data
smote = SMOTE(random_state=42)
X_train_norm_sm, Y_train_norm_sm = smote.fit_resample(X_train_norm, Y_train_norm)

#create the decision tree
norm_dec_tree = tree.DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')

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
          feature_names=X_sample.columns, 
          class_names=['<50k', ' >50k'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Decision Tree for Annual Salary Classification Without Feature Selection")
plt.show()

###CLASSIFICATION METHOD 1.5: DECISION TREES WITH FEATURE SELECTION

#fit SelectKBest to the original features
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X_sample, Y_sample)

#get the names of the selected features
selected_features = X_sample.columns[selector.get_support()]

print("Selected features:", selected_features)

#create a dataframe using only the selected features
adults_fs = pd.DataFrame()

for i in X_sample.columns[selector.get_support()]:
    adults_fs[i] = adults_encoded[i]

#create a new decision tree for the selected features
dec_tree_fs = tree.DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')

#split the attributes again
X_fs = X_sample
Y_fs = Y_sample

#split the data into test and train sets
X_train_fs, X_test_fs, Y_train_fs, Y_test_fs = train_test_split(
    X_fs, Y_fs,
    test_size = 0.4,
    stratify = Y_fs,
    random_state = 42
)
X_train_fs_sm, Y_train_fs_sm = smote.fit_resample(X_train_fs, Y_train_fs)


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

'''
plt.figure(figsize=(20, 10))
plot_tree(dec_tree_fs, 
          feature_names=X_fs.columns, 
          class_names=['edible', 'poisonous'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title("Decision Tree for Mushroom Classification With Feature Selection")
plt.show()'''

###CLASSIFICATION METHOD 2: NAIVE BAYES

#create the gaussian naive bayes for the normal dataframe
gb_norm = GaussianNB()

#train the model using the normal X and Y attributes
gb_norm.fit(X_train_norm_sm, Y_train_norm_sm)

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
gb_fs.fit(X_train_fs_sm, Y_train_fs_sm)

#predict on test and training sets
Y_NB_train_pred_fs = gb_fs.predict(X_train_fs)
Y_NB_test_pred_fs = gb_fs.predict(X_test_fs)

fs_gb_train_acc = accuracy_score(Y_train_fs, Y_NB_train_pred_fs)
fs_gb_test_acc = accuracy_score(Y_test_fs, Y_NB_test_pred_fs)

print(f"NB Train Acc With Feature Selection = {fs_gb_train_acc:.4f}") 
print(f"NB Test Acc With Feature Selection = {fs_gb_test_acc:.4f}", '\n')

###CLASSIFICATOIN METHOD 3: SUPPORT VECTOR MACHINES

#create the SVM for the normal attributes
svm_norm = SVC(class_weight='balanced')

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
svm_fs = SVC(class_weight='balanced')

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
knn_norm.fit(X_train_norm_sm, Y_train_norm_sm)

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
knn_fs.fit(X_train_fs_sm, Y_train_fs_sm)

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
mlp_norm.fit(X_train_norm_sm, Y_train_norm_sm)

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
mlp_fs.fit(X_train_fs_sm, Y_train_fs_sm)

#predict on the test and training sets
Y_MLP_train_pred_fs = mlp_fs.predict(X_train_fs)
Y_MLP_test_pred_fs = mlp_fs.predict(X_test_fs)

#calculate accuracies
fs_mlp_train_acc = accuracy_score(Y_train_fs, Y_MLP_train_pred_fs)
fs_mlp_test_acc = accuracy_score(Y_test_fs, Y_MLP_test_pred_fs)

print(f"MLP Train Acc With Feature Selection = {fs_mlp_train_acc:.4f}")
print(f"MLP Test Acc With Feature Selection = {fs_mlp_test_acc:.4f}", '\n')

###PLOTTING ALL ACCURACIES

#model names
models = ["Decision Tree", "Naive Bayes", "SVM", "KNN", "MLP"]

#accuracies without feature selection
train_acc_no_fs = [
    norm_tree_train_acc,
    norm_gb_train_acc,
    norm_svm_train_acc,
    norm_knn_train_acc,
    norm_mlp_train_acc
]

test_acc_no_fs = [
    norm_tree_test_acc,
    norm_gb_test_acc,
    norm_svm_test_acc,
    norm_knn_test_acc,
    norm_mlp_test_acc
]

#accuracies with feature selection
train_acc_fs = [
    fs_tree_train_acc,
    fs_gb_train_acc,
    fs_svm_train_acc,
    fs_knn_train_acc,
    fs_mlp_train_acc
]

test_acc_fs = [
    fs_tree_test_acc,
    fs_gb_test_acc,
    fs_svm_test_acc,
    fs_knn_test_acc,
    fs_mlp_test_acc
]

#plotting
x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(12, 6))
plt.bar(x - width, train_acc_no_fs, width, label='Train Acc (No FS)')
plt.bar(x, test_acc_no_fs, width, label='Test Acc (No FS)')
plt.bar(x + width, train_acc_fs, width, label='Train Acc (FS)')
plt.bar(x + 2*width, test_acc_fs, width, label='Test Acc (FS)')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison With and Without Feature Selection')
plt.xticks(x + width / 2, models)
plt.ylim(0.60, 0.9)
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.show()
