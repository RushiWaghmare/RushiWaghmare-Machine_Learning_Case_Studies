import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MarvellousHeadBrainPredictor():
    #Load data
    data=pd.read_csv("MarvellousHeadBrain.csv")

    print("Size of data set",data.shape)

    X=data['Head Size(cm^3)'].values  #tacking values in list formate
    Y=data['Brain Weight(grams)'].values
    #print(X)
    #print(Y)

    #Least Square method
    
    #mean of X
    total_X=0
    for i in X:
        total_X += i 
    mean_x=total_X/len(X)
    print("X Bar is : ",mean_x)

    #mean of Y
    total_Y=0
    for i in Y:
        total_Y += i 
    mean_y=total_Y/len(Y)
    print("Y Bar is : ",mean_y)

    #lengh of X
    n=len(X)

    numerator = 0
    denomenator = 0

    #Equation of line is y=mx + c
    print("Equation of line is y = mx + c ")
    for i in range(n):
        numerator=numerator + (X[i]-mean_x)*(Y[i]- mean_y)
        denomenator =denomenator + (X[i]-mean_x)**2 #Square

    # m = slope
    m = numerator/denomenator
    
    # c =  Value of y intersect , when x= 0 at x axis
    c = mean_y -(m * mean_x)

    print("Slope of Regression line (m) is ",m)
    print("Y intercept of Regression line (c) is",c)

    max_x = np.max(X)+100
    min_x = np.min(X)-100

    # Display potting of above points
    x = np.linspace(min_x,max_x,n)

    y = c + m*X

    plt.plot(x,y,color = '#58b970',label='Regression Line')

    plt.scatter(X,Y,color = '#ef5423',label='scatter point')

    plt.xlabel('Head size in cm^3')

    plt.ylabel('Brain weight in gram')

    plt.legend()
    plt.show()

    # Findout goodness of fit ie R Square
    ss_t=0
    ss_r=0

    for i in range(n):
        y_pred = c + m *X[i]
        ss_t += (Y[i] - mean_y)**2
        ss_r += (Y[i] - y_pred)**2

    r2= 1-(ss_r/ss_t)
    print(r2)

def main():

    print("Suervised Machine Learning")

    print("Linear Regression on head and Brain Size data set")

    MarvellousHeadBrainPredictor()

if __name__ == "__main__":
    main()