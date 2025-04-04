import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC  # Not used here, but can be added if desired
import seaborn as sns

# ---------------------------
# 1. Load and Preprocess the Data
# ---------------------------
adults = pd.read_csv('./adult.data',
                     names=['age','workclass','fnlwgt', 'education',
                            'education-num', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'capital-gain',
                            'capital-loss', 'hours-per-week', 'native-country','annual-income'])

# Display the first few rows and general info
print(adults.head())
print(adults.info())

# Summary statistics for categorical features (using value_counts)
for column in adults.columns:
    print(f"\nValue counts for {column}:\n", adults[column].value_counts())

# Check for missing values
missing_values = adults.isnull().sum()
print("\nMissing values in each column:\n", missing_values)

# Visualize the class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='annual-income', data=adults)
plt.title("Distribution of Adult Classes (>50 000 vs <=50 000)")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Replace '?' with NaN and drop rows with missing values
adults = adults.replace('?', np.nan).dropna()

# One-hot encode selected categorical features
features_to_encode = ['workclass','education','marital-status',
                      'occupation','relationship','race','sex',
                      'native-country']
adults_encoded = pd.get_dummies(adults, columns=features_to_encode, drop_first=True)

# ---------------------------
# 2. Prepare Data for Modeling
# ---------------------------
X_full = adults_encoded.drop(['annual-income'], axis=1)
Y_full = adults_encoded['annual-income']

# For consistent results, convert target labels to a binary numeric format:
# 1 for '>50K', 0 for '<=50K'
Y_full_bin = Y_full.str.strip().apply(lambda x: 1 if x == '>50K' else 0)

# Take a stratified sample of 10,000 instances
X_sample, _, Y_sample, _ = train_test_split(X_full, Y_full_bin,
                                            train_size=10000,
                                            stratify=Y_full_bin,
                                            random_state=42)

# Split sample into training (70%) and test sets (30%)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_sample, Y_sample,
    test_size=0.30,
    stratify=Y_sample,
    random_state=42
)

# ---------------------------
# 3. Apply SMOTE to Training Data
# ---------------------------
smote = SMOTE(random_state=42)
X_train_sm, Y_train_sm = smote.fit_resample(X_train, Y_train)

# ---------------------------
# 4. Train Models
# ---------------------------
# 4a. Decision Tree (using class weighting)
dt_model = tree.DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
dt_model.fit(X_train, Y_train)

# 4b. Gaussian Naive Bayes (trained on SMOTE data)
gnb_model = GaussianNB()
gnb_model.fit(X_train_sm, Y_train_sm)

# ---------------------------
# 5. Generate Predictions & Evaluate
# ---------------------------
# Decision Tree predictions and probabilities
Y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

# Gaussian NB predictions and probabilities
Y_pred_gnb = gnb_model.predict(X_test)
y_prob_gnb = gnb_model.predict_proba(X_test)[:, 1]

# Print accuracies
print("Decision Tree Accuracy: {:.4f}".format(accuracy_score(Y_test, Y_pred_dt)))
print("Gaussian NB Accuracy: {:.4f}".format(accuracy_score(Y_test, Y_pred_gnb)))

# ---------------------------
# 6. Plot ROC Curves on Separate Graphs
# ---------------------------
# ROC for Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(Y_test, y_prob_dt, pos_label=1)
roc_auc_dt = auc(fpr_dt, tpr_dt)
plt.figure(figsize=(8, 6))
plt.plot(fpr_dt, tpr_dt, color='blue', lw=2, label=f'Decision Tree ROC (AUC = {roc_auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ROC for Gaussian NB
fpr_gnb, tpr_gnb, _ = roc_curve(Y_test, y_prob_gnb, pos_label=1)
roc_auc_gnb = auc(fpr_gnb, tpr_gnb)
plt.figure(figsize=(8, 6))
plt.plot(fpr_gnb, tpr_gnb, color='green', lw=2, label=f'Gaussian NB ROC (AUC = {roc_auc_gnb:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gaussian Naive Bayes')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ---------------------------
# 7. Plot Confusion Matrices on Separate Graphs
# ---------------------------
# Decision Tree Confusion Matrix
cm_dt = confusion_matrix(Y_test, Y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=['<=50K', '>50K'])
plt.figure(figsize=(8, 6))
disp_dt.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# Gaussian NB Confusion Matrix
cm_gnb = confusion_matrix(Y_test, Y_pred_gnb)
disp_gnb = ConfusionMatrixDisplay(confusion_matrix=cm_gnb, display_labels=['<=50K', '>50K'])
plt.figure(figsize=(8, 6))
disp_gnb.plot(cmap=plt.cm.Greens)
plt.title('Confusion Matrix - Gaussian Naive Bayes')
plt.show()
