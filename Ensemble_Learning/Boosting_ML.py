#MNIST case study
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

data=pd.read_csv('mnist.csv')

df_x = data.iloc[:,1:]
df-y = data.iloc[:,0]

x_train,x_test,y_train,y_test =train_test_split(df_x,df_y,test_size=0.2,random_state=4)

obj= DecisionTressClassifier(__,__,__);
adv=AdaBoostClassifier(obj,n_estimators = __,learning_rate = __);
