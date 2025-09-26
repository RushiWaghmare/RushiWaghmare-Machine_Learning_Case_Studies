import pandas as pd
from sklearn.linear_model import LinearRegression

#stpe 1: load data
def load_brainData():
    
    data=pd.read_csv("MarvellousHeadBrain.csv")
    print(data.head(5))
    return data

#step 2: reshape the datset
def reshape_brainData(data):

    X=data['Head Size(cm^3)'].values 
    Y=data['Brain Weight(grams)'].values
    
    
    # Reshape the data in Virtical Column
    print(X)
    X=X.reshape((-1,1))
    print("Size of data set : ",data.shape)
    print(X)
    return X,Y

#step 3: Algorithm selection 
def Algo_brainData():

    reg= LinearRegression()
    return reg

#step 4: Algorithm train
def AlgoTrain_brainData(reg,X,Y):

    reg = reg.fit(X,Y)

    return reg,X,Y

#step 5: Algorithm test
def AlgoTest_brainData(reg,X,Y):

    y_pred = reg.predict(X)

    r2 = reg.score(X,Y)

    print("Godess of fit : ",r2)



    

def main():
    Data=load_brainData()
    X,Y=reshape_brainData(Data)
    reg=Algo_brainData()
    train=AlgoTrain_brainData(reg,X,Y)
    result= AlgoTest_brainData(reg,X,Y)

if __name__ == "__main__":
    main()
