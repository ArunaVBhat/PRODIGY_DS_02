import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_df = pd.read_csv(r"C:\Users\91948\Downloads\titanic.csv")
titanic_df.head()
missing_values = titanic_df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Handle missing values
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df.dropna(subset=['Embarked'], inplace=True)
print(titanic_df.dtypes)
titanic_df['Survived'] = titanic_df['Survived'].astype('category')
# Basic statistics
print(titanic_df.describe())

# Visualize the distribution of passengers by survival
sns.countplot(x='Survived', data=titanic_df)
plt.title('Count of Passengers by Survival Status')
plt.show()

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(titanic_df['Age'], bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Explore the relationship between 'Pclass' and 'Survived'
sns.countplot(x='Pclass', hue='Survived', data=titanic_df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Explore the relationship between 'Sex' and 'Survived'
sns.countplot(x='Sex', hue='Survived', data=titanic_df)
plt.title('Survival Rate by Gender')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(titanic_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
