import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_and_visualize_data():
    # Load Titanic dataset from seaborn
    titanic_data = sns.load_dataset("titanic")
    print("Top 5 rows of dataset:")
    print(titanic_data.head())

    # Data Preprocessing - Handle missing values
    titanic_data.drop(["deck", "embark_town", "alive", "class", "who", "adult_male"], axis=1, inplace=True)
    titanic_data.dropna(inplace=True)

    # Visualizations
    sns.set_style("whitegrid")

    # Plot Survival Count
    plt.figure(figsize=(6, 4))
    sns.countplot(data=titanic_data, x="survived", palette="coolwarm")
    plt.title("Survival Count")
    plt.show()

    # Plot Survival by Gender
    plt.figure(figsize=(6, 4))
    sns.countplot(data=titanic_data, x="sex", hue="survived", palette="viridis")
    plt.title("Survival Count by Gender")
    plt.show()

    # Plot Age Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(titanic_data["age"], bins=30, kde=True, color="blue")
    plt.title("Age Distribution")
    plt.show()

    return titanic_data


def preprocess_data(data):
    # Convert categorical data
    data = pd.get_dummies(data, columns=["sex", "embarked"], drop_first=True)
    
    # Drop irrelevant columns
    data.drop(["sibsp", "parch"], axis=1, inplace=True)
    
    # Split features and target
    X = data.drop("survived", axis=1)
    y = data["survived"]
    
    return X, y


def train_and_evaluate_model(X, y):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:\n", conf_matrix)
    
    # Classification Report
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def main():
    # Step 1: Load & visualize data
    data = load_and_visualize_data()
    
    # Step 2: Preprocess data
    X, y = preprocess_data(data)
    
    # Step 3: Train and evaluate model
    train_and_evaluate_model(X, y)


if __name__ == "__main__":
    main()
