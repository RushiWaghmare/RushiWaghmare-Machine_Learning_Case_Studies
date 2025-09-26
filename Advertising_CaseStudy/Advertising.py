import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
def Advertising_sellsPredictor(path):
    data=pd.read_csv(path )
    print(data.head())

    x=data.drop("sales",axis=1)
    y=data.sales

    #split Dataset
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=42)

    #train algorithms
    lr=LinearRegression()
    lr.fit(x_train,y_train)

    y_pred=lr.predict(x_test)

    print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

    


def main():
    Advertising_sellsPredictor("AdvertisingData.csv")
if __name__ == "__main__":
    main()