import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print("----Diabetes predictor using K Nearest neighbour-----------")

data=pd.read_csv('diabetes.csv')

print("Column of Dataset")
print(data.columns)
print("Top 5 entries of Diabetes Data : ")
print(data.head())
print("Dimension of Diabetes Data : {}".format(data.shape))

x=data.drop("Outcome",axis=1)
y=data["Outcome"]

x_train ,x_test, y_train, y_test = train_test_split(x,y,stratify =data["Outcome"],random_state=66)

training_accuracy = []
test_accuracy = []

#try n_neighbors form 1 to 10
#print("Accuracy of Training and Testing of Dataset in range(1 to 10)")
neighbors_settings = range (1,11)

for n_neighbors in neighbors_settings:

    #bulid model
    knn =KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train , y_train)

    # record training
    training_accuracy.append(knn.score(x_train,y_train))

    #record test set accuracy
    test_accuracy.append(knn.score(x_test, y_test))

#print(training_accuracy)
#print(test_accuracy)

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")

plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")

plt.legend()
plt.savefig('knn_compare_model')
plt.show()

knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(x_train, y_train)

print("Accuracy of K-NN classifier on test set: {:.2f}".format(knn.score(x_test,y_test)))





