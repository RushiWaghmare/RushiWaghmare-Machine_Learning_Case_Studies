import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

print("----Diabetes predictor using Decision Tree-----------")

data=pd.read_csv('diabetes.csv')

print("Column of Dataset")
print(data.columns)
print("Top 5 entries of Diabetes Data : ")
print(data.head())
print("Dimension of Diabetes Data : ")
print(format(data.shape))

x=data.drop("Outcome",axis=1)
y=data["Outcome"]

x_train ,x_test, y_train, y_test = train_test_split(x,y,stratify =data["Outcome"],random_state=66)

tree=DecisionTreeClassifier(random_state=0)

tree.fit(x_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(x_train,y_train)))

print("Accuracy on testing set: {:.3f}".format(tree.score(x_test,y_test)))

tree=DecisionTreeClassifier(max_depth=3,random_state=0)
tree.fit(x_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(x_train,y_train)))

print("Accuracy on testing set: {:.3f}".format(tree.score(x_test,y_test)))

print("Feature importance:\n{}".format(tree.feature_importances_))

#ploting

def plot_feature_importances_diabetes(model): 
 plt.figure(figsize=(8,6)) 
 n_features = 8 
 plt.barh(range(n_features), model.feature_importances_, align='center') 
 diabetes_features = [x for i,x in enumerate(data.columns) if i!=8] 
 plt.yticks(np.arange(n_features), diabetes_features) 
 plt.xlabel("Feature importance") 
 plt.ylabel("Feature") 
 plt.ylim(-1, n_features) 
 plt.show() 

plot_feature_importances_diabetes(tree)