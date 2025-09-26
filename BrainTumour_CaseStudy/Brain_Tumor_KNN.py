#################################################
#import required modules
#################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#################################################
# Function_Name:
# Discription: 
# Input: 
# Output: 
# Date: 09/08/2024
# Author: Rushikesh Waghmare
#################################################

#################################################
# Function_Name: Data_read()
# Discription: 
# Input: CSV file
# Output: Stores data in variable called "data"
# Date: 09/08/2024
# Author: Rushikesh Waghmare
#################################################
def Data_read(path):
    Data=pd.read_csv(path)
    print(Data.head())
    return Data

#################################################
# Function_Name: Data_Initialise():
# Discription: Split Data in Dependent Variables and Independent Variables
# Input:  Data
# Output: Divided Data
# Date: 09/08/2024
# Author: Rushikesh Waghmare
#################################################
def Data_Initialise(Data):
    label_encoder = LabelEncoder()

    Data["Location"]=label_encoder.fit_transform(Data["Location"])
    print(Data["Location"])
    Data["Grade"]=label_encoder.fit_transform(Data["Grade"])
    print( Data["Grade"])
    Data["Gender"]=label_encoder.fit_transform(Data["Gender"])
    print(Data["Gender"])
    X=Data[["Gender","Grade","Location","Size (cm)","Patient_Age"]]
    Y=Data["Tumor_Type"]
    print(X)
    print(Y)
    return X,Y


#################################################
# Function_Name: Data_Train()
# Discription: Split Data in Dependent Variables and Independent Variables
# Input:  Data
# Output: Divided Data
# Date: 09/08/2024
# Author: Rushikesh Waghmare
#################################################
def Data_Train(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.8,random_state=42)
    return  X_train, X_test, Y_train, Y_test 


#################################################
# Function_Name: 
# Discription: 
# Input: 
# Output: 
# Date: 09/08/2024
# Author: Rushikesh Waghmare
#################################################
def DecisionTree_Classifier( X_train, X_test, Y_train, Y_test ):
    DT=DecisionTreeClassifier()
    DT.fit(X_train,Y_train)
    Y_pred=DT.predict(X_test)
    print("The Testing Accuracy : ",accuracy_score(Y_test,Y_pred)*100,"%")

#################################################
# Function_Name: 
# Discription: 
# Input: 
# Output: 
# Date: 09/08/2024
# Author: Rushikesh Waghmare
#################################################
def Logistic_Reggression(X_train, X_test, Y_train, Y_test ):
    LR=LogisticRegression (max_iter=1000)
    LR.fit(X_train,Y_train)
    Y_pred=LR.predict(X_test)
    print("The Testing Accuracy : ",accuracy_score(Y_test,Y_pred)*100,"%")


def RandomForest_Classifier(X_train, X_test, Y_train, Y_test ):
    RF=RandomForestClassifier(n_estimators=100)
    RF.fit(X_train,Y_train)
    Y_pred=RF.predict(X_test)
    print("The Testing Accuracy : ",accuracy_score(Y_test,Y_pred)*100,"%")


def main():
    Data=Data_read("brain_tumor_dataset.csv")
    X,Y=Data_Initialise(Data)
    X_train, X_test, Y_train, Y_test  = Data_Train(X,Y)
   # DecisionTree_Classifier(X_train, X_test, Y_train, Y_test)
    #Logistic_Reggression(X_train, X_test, Y_train, Y_test)
    RandomForest_Classifier(X_train, X_test, Y_train, Y_test )

if __name__ == "__main__":
    main()