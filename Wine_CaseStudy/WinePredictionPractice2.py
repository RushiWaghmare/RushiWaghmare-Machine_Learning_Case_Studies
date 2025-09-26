import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def WinePredictor():

    #import data form CSV file
    data=pd.read_csv("WinePredictorFile.csv")

    
    """label_encoder=LabelEncoder()
    #fetures=['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
    data['Class']=label_encoder.fit_transform(data['Class'])

    X=data[['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']]
    Y= data['Class']"""

    target = 'Class'
    fetures = data.columns.drop(target)

    X= data[fetures].values
    Y= data[target].values

    # scale fetures
    scaler = StandardScaler()
    x_scaled= scaler.fit_transform(X)

    #split data
    X_train,X_test, Y_train, Y_test = train_test_split(x_scaled,Y,test_size=0.3)

    #create classifier
    knn=KNeighborsClassifier(n_neighbors=3)

    #train
    knn.fit(X_train, Y_train)

    #test
    y_pred=knn.predict(x_test)
    print(y_pred)




def main():
    print("Wine Prediction using K Neighbors Classification Algorithm")
    WinePredictor()
if __name__ == "__main__":
    main()