import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('./agaricus-lepiota.data',
                        names=['classification','cap-shape','cap-surface','cap-color',
                               'bruises?','odor','gill-attachment',
                               'gill-spacing','gill-size','gill-color',
                               'stalk-shape','stalk-root','stalk-surface-above-ring',
                               'stalk-surface-below-ring','stalk-color-above-ring',
                               'stalk-color-below-ring','veil-type',
                               'veil-color','ring-number','ring-type',
                               'spore-print-color','population','habitat'])  # Ensure the file path is correct

# Display the first few rows and general info
print(data.head())
print(data.info())

# Summary statistics for categorical features (using value_counts)
for column in data.columns:
    print(f"\nValue counts for {column}:\n", data[column].value_counts())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values in each column:\n", missing_values)

# Visualize the class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='classification', data=data)
plt.title("Distribution of Mushroom Classes (Edible vs Poisonous)")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Visualize one of the categorical features against the class, e.g., 'cap-shape'
plt.figure(figsize=(8, 5))
sns.countplot(x='cap-shape', hue='class', data=data)
plt.title("Cap Shape Distribution by Mushroom Class")
plt.xlabel("Cap Shape")
plt.ylabel("Frequency")
plt.show()
