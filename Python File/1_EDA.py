# Code 2

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('/content/titanic.csv')

### Basic Data Overview

df.head()

print(df.describe())

print(df.info())

### Pair Plot for Titanic Dataset

sns.pairplot(df, hue='Survived', palette={0: 'red', 1: 'green'}, diag_kind='kde')
plt.title('Pair Plot of Features Colored by Survival')
plt.show()

### Handling Missing Values

print("Missing Values Before Handling:\n", df.isna().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df.drop(columns=['Cabin'], inplace=True)  # Drop sparse column

print("\nMissing Values After Handling:\n", df.isna().sum())

### Feature Engineering

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df['IsAlone'] = (df['FamilySize']==1).astype(int)

df.drop(columns=['SibSp', 'Parch'])

### Drop Irrelevant Columns

df.drop(columns=['Name', 'Ticket'], inplace=True)

### Encoding Categorical Variables

from sklearn.preprocessing import LabelEncoder

df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

### Correlation Matrix

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

### Feature Scaling

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

### Final Dataset Overview

print(df.head())

print(df.describe())

print(df.info())