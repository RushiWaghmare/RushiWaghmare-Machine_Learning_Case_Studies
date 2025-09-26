import sklearn 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def Accuracy_play():
    data=pd.read_csv("PlayPredict_File.csv")
    print("Data before Encoding:")
    print(data)
    #Encoding
    label_encoder=LabelEncoder()

    #Whether : overcast=0,rainy =1,sunny=2
    data['Weather']=label_encoder.fit_transform(data['Weather'])
    #Temprature : cool =0 ,Hot=1 , mild =2 , 
    data['Temprature']=label_encoder.fit_transform(data['Temprature'])
    #Play : yes =1 , No = 0
    data['Play']= label_encoder.fit_transform(data['Play'])
    print("Data after encoding:")
    print(data[['Weather','Temprature','Play']])
    
    
    X =data[['Weather','Temprature']]
    Y =data['Play']

    #spliting data
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3,random_state=42)

    #train algorithm
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(train_X, train_Y)

    #test algorithm
    result =classifier.predict([[2,0]])
    print(result)

    accuracy=classifier.score(test_X,test_Y)
    print("Accuracy of algorithms: ",accuracy)

def main():
    Accuracy_play()

if __name__ == "__main__":
    main()