# Regression predifined
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
def Read_Data(Path):
    Data = pd.read_csv(Path)
    print(Data.head(5))
    print(Data.info())
    return Data

def Data_initialise(Data):
    X = Data["carmodel"].values
    Y = Data["price"].values  
    X=X.reshape((-1,1))
    print(f"Size of Data set :v {Data.shape}")
    return X,Y 

def Algo_object():
    Reg = LinearRegression()
    return Reg

def Algo_train(Reg,X,Y):

    Reg = Reg.fit(X,Y)
    return Reg,X,Y

def Algo_test( Reg,X,Y):
    Y_pred = Reg.predict(X)
    R2 = Reg.score(X,Y)
    print(R2)

def main():

    Data = Read_Data("Cars_Data.csv")

    X,Y = Data_initialise(Data)
    Reg = Algo_object()
    Reg,X,Y = Algo_train(Reg,X,Y)
    Algo_test( Reg,X,Y)

if __name__ == "__main__":
    main()
