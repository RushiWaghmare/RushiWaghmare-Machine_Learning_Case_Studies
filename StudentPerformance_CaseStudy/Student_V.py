# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('Student_performance_data _.csv')  # Replace with the actual path to your dataset

# Data Preprocessing (Add your preprocessing steps here)

# Visualization 1: Distribution of GradeClass
plt.figure(figsize=(8, 6))
sns.countplot(x='GradeClass', data=df, palette='Set2')
plt.title('Distribution of GradeClass')
plt.xlabel('Grade Class')
plt.ylabel('Count')
plt.show()

# Visualization 2: Correlation Heatmap of Numeric Features
plt.figure(figsize=(10, 8))
numeric_columns = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']  # Add more numeric columns if available
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# Visualization 3: GPA Distribution by Parental Education
plt.figure(figsize=(10, 6))
sns.boxplot(x='ParentalEducation', y='GPA', data=df, palette='Set3')
plt.title('GPA Distribution by Parental Education')
plt.xlabel('Parental Education')
plt.ylabel('GPA')
plt.xticks(rotation=45)
plt.show()

# Visualization 4: Study Time vs GPA Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='StudyTimeWeekly', y='GPA', hue='GradeClass', data=df, palette='Set1')
plt.title('Study Time vs GPA')
plt.xlabel('Study Time (Weekly)')
plt.ylabel('GPA')
plt.show()

# Visualization 5: Countplot of Extracurricular Activities Participation
plt.figure(figsize=(8, 6))
sns.countplot(x='Extracurricular', hue='GradeClass', data=df, palette='pastel')
plt.title('Participation in Extracurricular Activities by Grade Class')
plt.xlabel('Participates in Extracurricular')
plt.ylabel('Count')
plt.legend(title='Grade Class')
plt.show()

# Visualization 6: Parental Support vs GradeClass
plt.figure(figsize=(8, 6))
sns.countplot(x='ParentalSupport', hue='GradeClass', data=df, palette='coolwarm')
plt.title('Parental Support and Grade Class')
plt.xlabel('Parental Support')
plt.ylabel('Count')
plt.legend(title='Grade Class')
plt.show()

