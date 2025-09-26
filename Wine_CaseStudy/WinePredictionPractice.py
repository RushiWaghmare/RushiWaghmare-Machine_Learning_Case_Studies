from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictor():
    wine = datasets.load_wine()

    #print names of the features and targets or lable names
    print(wine.feature_names)
    print(wine.target_names)

    #print wine data top 5
    #print(wine.data[0:5])

    #print the wine lables 
    #print(wine.target)

    #split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(wine.data, wine.target,test_size=0.3)

    #create classifier
    knn=KNeighborsClassifier(n_neighbors=3)

    #train the algorithm
    knn.fit(X_train, Y_train)
    
    #test the algorithms  result= y_prediction
    result=knn.predict(X_test)

    print(result)
    
    #Calculate accuracy using matrics
    accuracy=metrics.accuracy_score(Y_test,result)
    print("Accuracy of algortihm is : ",accuracy *100,"%")




def  main():
    print("Wine Predictor application using K Nearest Neighbor algoritm")
    WinePredictor()
if __name__ == "__main__":
    main()