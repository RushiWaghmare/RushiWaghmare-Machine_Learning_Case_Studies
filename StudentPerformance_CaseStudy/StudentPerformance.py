#################################################
# Import all required modules
#################################################
from sklearn.tree import DecisionTreeClassifier
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns

#################################################
# Function_Name: GetData()
# Description: Given function used to store data from csv file in variable
# Input: CSV file
# Output: Stores data in variable called "data"
# Date: 29/07/2024
# Author: Rushikesh Waghmare
#################################################
def GetData(Path):
    Data = pd.read_csv(Path)
    print("Top 5 entries of data: ", Data.head(5))
    return Data

#################################################
# Function_Name: ShowData()
# Description: 
# Input: CSV file
# Output: Stores data in variable called "data"
# Date: 29/07/2024
# Author: Rushikesh Waghmare
#################################################
def ShowData(Data):
    print("Top 5 entries of data: ", Data.head(5))
    return Data

def Data_types(Data):
    print(Data.dtypes)
    print("\n")
    print(Data.info())

def DataInitialise(Data):
    X = Data.drop('GradeClass', axis=1)
    Y = Data['GradeClass']  
    return X, Y

def DataManipulation(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test

#################################################
# Visualization Function
#################################################
def Visualisation(Data):
    # 1. Distribution of GradeClass
    plt.figure(figsize=(8, 6))
    sns.countplot(x='GradeClass', data=Data, palette='Set2')
    plt.title('Distribution of GradeClass')
    plt.xlabel('Grade Class')
    plt.ylabel('Count')
    plt.show()

    # 2. Study Time vs GPA Scatter Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='StudyTimeWeekly', y='GPA', hue='GradeClass', data=Data, palette='Set1')
    plt.title('Study Time vs GPA')
    plt.xlabel('Study Time (Weekly)')
    plt.ylabel('GPA')
    plt.show()

    # 3. Countplot of Extracurricular Activities Participation
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Extracurricular', hue='GradeClass', data=Data, palette='pastel')
    plt.title('Participation in Extracurricular Activities by Grade Class')
    plt.xlabel('Participates in Extracurricular')
    plt.ylabel('Count')
    plt.legend(title='Grade Class')
    plt.show()

    # 4. Parental Support vs GradeClass
    plt.figure(figsize=(8, 6))
    sns.countplot(x='ParentalSupport', hue='GradeClass', data=Data, palette='coolwarm')
    plt.title('Parental Support and Grade Class')
    plt.xlabel('Parental Support')
    plt.ylabel('Count')
    plt.legend(title='Grade Class')
    plt.show()

    # 5. Age Histogram
    plt.hist(Data["Age"], bins=8, color="green")
    plt.title("Divided into Age")
    plt.xlabel("Count")
    plt.ylabel("Y axis")
    plt.show()

    # 6. Gender Histogram
    plt.hist(Data["Gender"], bins=2, color="g")
    plt.title("Gender")
    plt.xlabel("Male and Female")
    plt.ylabel("Count")
    plt.show()

#################################################
# Model selection functions
#################################################
def DecisionTree_Classifier(X_train, X_test, Y_train, Y_test):
    Dt = DecisionTreeClassifier(random_state=66)
    Dt.fit(X_train, Y_train)
    Y_pred = Dt.predict(X_test)
    print("Accuracy using DecisionTreeClassifier Algorithm {:.2f}%".format(accuracy_score(Y_test, Y_pred)*100))

def KNN_Classifier(X_train, X_test, Y_train, Y_test):
    Knn = KNeighborsClassifier(n_neighbors=3)
    Knn.fit(X_train, Y_train)
    Y_pred = Knn.predict(X_test)
    print("Accuracy using KNeighborsClassifier Algorithm {:.2f}%".format(accuracy_score(Y_test, Y_pred)*100))

def LogisticAlgo_Classifier(X_train, X_test, Y_train, Y_test):
    LR = LogisticRegression(max_iter=1000)
    LR.fit(X_train, Y_train)
    Y_pred = LR.predict(X_test)
    print("Accuracy using LogisticRegression Algorithm {:.2f}%".format(accuracy_score(Y_test, Y_pred)*100))

def RandomForest_Classifier(X_train, X_test, Y_train, Y_test):
    RF = RandomForestClassifier(n_estimators=100)
    RF.fit(X_train, Y_train)
    Y_pred = RF.predict(X_test)
    print("Accuracy using RandomForestClassifier Algorithm {:.2f}%".format(accuracy_score(Y_test, Y_pred)*100))

def RandomForest_Classifier2(X_train, X_test, Y_train, Y_test):
    RF = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
    RF.fit(X_train, Y_train)
    Y_pred = RF.predict(X_test)
    print("Accuracy using RandomForestClassifier Algorithm with Max Depth 3 is {:.2f}%".format(accuracy_score(Y_test, Y_pred)*100))

def LinearRegression_algorithm(X_train, X_test, Y_train, Y_test):
    LR = LinearRegression()
    LR.fit(X_train, Y_train)
    Y_pred = X_test
    print("Mean Squared Error using LinearRegression {:.2f}".format(metrics.mean_squared_error(Y_test, Y_pred)))

#################################################
# Function_Name: main()
# Description: Entry point function to call other functions
#################################################
def main():
    Data = GetData("Student_performance_data _.csv")
    Data_types(Data)
    X, Y = DataInitialise(Data)
    X_train, X_test, Y_train, Y_test = DataManipulation(X, Y)
    Visualisation(Data)
    DecisionTree_Classifier(X_train, X_test, Y_train, Y_test)
    # KNN_Classifier(X_train, X_test, Y_train, Y_test)
    # LogisticAlgo_Classifier(X_train, X_test, Y_train, Y_test)
    RandomForest_Classifier(X_train, X_test, Y_train, Y_test)
    # RandomForest_Classifier2(X_train, X_test, Y_train, Y_test)
    # LinearRegression_algorithm(X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":
    main()
