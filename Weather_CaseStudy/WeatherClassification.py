import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
from sklearn.ensemble import RandomForestClassifier,VotingClassifier

def Read_Data(Path):
    Data=pd.read_csv(Path)
    print(Data.head(5))
    print(Data.info())
    print(Data.dtypes)
    return Data

def Data_Encoding(Data):
    Label_Encoder=LabelEncoder()
    Data["Cloud Cover"]=Label_Encoder.fit_transform(Data["Cloud _Cover"])
    #print(Data["Cloud Cover"])
    Data["Season"]=Label_Encoder.fit_transform(Data["Season"])
    Data["Location"]=Label_Encoder.fit_transform(Data["Location"])
    X=Data[["Cloud Cover","Season","Location","Temperature","Humidity","Wind Speed","Precipitation (%)","Atmospheric Pressure","UV Index","Visibility (km)"]]
    Y=Data["Weather Type"]
    return X,Y

def Feature_Scaling(X,Y):
    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(X)
    print(Y)
    return X, Y

def Data_Train(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def DecisionTree_Classifier(X_train, X_test, Y_train, Y_test):
    DT=DecisionTreeClassifier()
    DT.fit(X_train,Y_train)
    Y_pred=DT.predict(X_test)
    print("Testing Accuracy using DecisionTree_Classifier: ",accuracy_score(Y_test,Y_pred))
    print("Confusion Metrix: ")
    print(confusion_matrix(Y_test,Y_pred))
    print("Classification Report: ")
    print(classification_report(Y_test,Y_pred))
    #print("F1 Score: ",f1_score(Y_test,Y_pred))
    return DT

def Logistic_reggression(X_train, X_test, Y_train, Y_test):
    LR=LogisticRegression(max_iter=1000)
    LR.fit(X_train,Y_train)
    Y_pred=LR.predict(X_test)
    print("Testing Accuracy using Logistic_reggression: ",accuracy_score(Y_test,Y_pred))
    print("Confusion Metrix: ")
    print(confusion_matrix(Y_test,Y_pred))
    print("Classification Report: ")
    print(classification_report(Y_test,Y_pred))
    return LR 

def RandomForest_Classifier(X_train, X_test, Y_train, Y_test):
    RF=RandomForestClassifier(n_estimators=1000)
    RF.fit(X_train,Y_train)
    Y_pred=RF.predict(X_test)
    print("Testing Accuracy using RandomForest_Classifier: ",accuracy_score(Y_test,Y_pred))
    print("Confusion Metrix: ")
    print(confusion_matrix(Y_test,Y_pred))
    print("Classification Report: ")
    print(classification_report(Y_test,Y_pred))
    return RF

def Voting_Classifier(X_train, X_test, Y_train, Y_test):
    log_clf = LogisticRegression(max_iter=1000)
    rnd_clf = RandomForestClassifier(n_estimators=1000)
    dt_clf = DecisionTreeClassifier()

    vot_clf = VotingClassifier(estimators=[
        ('lr', log_clf), 
        ('rf', rnd_clf),
        ('dt', dt_clf)
    ], voting='hard')

    vot_clf.fit(X_train,Y_train)

    Y_pred = vot_clf.predict(X_test)

    print("Testing Accuracy Using Voting Classifier:",accuracy_score(Y_test,Y_pred)*100)
    print(confusion_matrix(Y_test,Y_pred))
    print("Classification Report: ")
    print(classification_report(Y_test,Y_pred))


def main():
    Data=Read_Data("weather_classification_data.csv")
    X,Y= Data_Encoding(Data)
    X,Y=Feature_Scaling(X,Y)
    X_train, X_test, Y_train, Y_test= Data_Train(X,Y)
    #DecisionTree_Classifier(X_train, X_test, Y_train, Y_test)
    #Logistic_reggression(X_train, X_test, Y_train, Y_test)
    #RandomForest_Classifier(X_train, X_test, Y_train, Y_test)
    Voting_Classifier(X_train, X_test, Y_train, Y_test)
if __name__ == "__main__":
    main()