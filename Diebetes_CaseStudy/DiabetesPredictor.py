##############################################################
"""import required mdules"""
############################################################
##############################################################
"""
Function Name :  
Description : 
Input :  
Output : 
Developer : Rushikesh Ratnakar Waghmare
Date : 19/07/2024
"""
##############################################################
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##############################################################
"""
Function Name :  Read()
Description : Read the data from given CSV file
Input :  File path
Output : File data
Developer : Rushikesh Ratnakar Waghmare
Date : 19/07/2024
"""
##############################################################
def Read(path):
    data=pd.read_csv(path)
    print(data.head(5))
    return data

##############################################################
"""
Function Name :  
Description : 
Input :  
Output : 
Developer : Rushikesh Ratnakar Waghmare
Date : 19/07/2024
"""
##############################################################
def Initialise_data(data):
    x=data.drop('Outcome',axis=1)
    y=data['Outcome']
    return x,y

##############################################################
"""
Function Name :  
Description : 
Input :  
Output : 
Developer : Rushikesh Ratnakar Waghmare
Date : 19/07/2024
"""
##############################################################
def Manipulation_data(x,y):
    x_train, x_test, y_train, y_test= train_test_split(x,y,test_size =0.2,random_state=66)
    return x_train, x_test, y_train, y_test

def KNN_algo(x_train, x_test, y_train, y_test):
    Knn=KNeighborsClassifier(n_neighbors=3)
    Knn.fit(x_train, y_train)
    y_pred= Knn.predict(x_test)
    #training_accuracy=accuracy_score(x_train, y_train)
    testing_accuracy=accuracy_score(y_test,y_pred)
   # print("Training Accuracy: ",training_accuracy)
    print("Testing Accuracy using K-NeighborsClassifier Algorithm: ",testing_accuracy)
    

def Random_algo(x_train, x_test, y_train, y_test):
    rf=RandomForestClassifier(n_estimators=100,random_state=0)
    rf.fit(x_train, y_train)
    y_pred=rf.predict(x_test)
    testing_accuracy= accuracy_score(y_test, y_pred)
    print("Testing Accuracy using RandomForestClassifier Algorithm: ",testing_accuracy)

def Random2_algo(x_train, x_test, y_train, y_test):
    rf=RandomForestClassifier(max_depth=2,n_estimators=100,random_state=0)
    rf.fit(x_train, y_train)
    y_pred=rf.predict(x_test)
    testing_accuracy= accuracy_score(y_test, y_pred)
    print("Testing Accuracy with depth 2 using RandomForestClassifier Algorithm: ",testing_accuracy)

##############################################################
"""
Function Name :  main()
Description : calling all functions
Input :  File path
Output : respective output on functions
Developer : Rushikesh Ratnakar Waghmare
Date : 19/07/2024
"""
##############################################################
def main():
    data=Read("diabetes.csv")
    x,y=Initialise_data(data)
    x_train, x_test, y_train, y_test = Manipulation_data(x,y)
    KNN_algo(x_train, x_test, y_train, y_test)
    Random_algo(x_train, x_test, y_train, y_test)
    Random2_algo(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
