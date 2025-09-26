import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


data= pd.read_csv('PlayPredict_File.csv')
label_encoder=LabelEncoder()

print("Data before encoding: ")
print(data)

data['Wether']=label_encoder.fit_transform(data['Wether'])
data['Temprature']=label_encoder.fit_transform(data['Temprature'])
data['Play']=label_encoder.fit_transform(data['Play'])

print("Data after encoding: ")
print(data)


X=['Wether','Temprature']
Y=['Play']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

classifier = KNeighborsClassifier(n_neighbors=5)

#Train algorithm
classifier=classifier.fit(X,Y)

res=classifier.predict([[2,0]])
print(res)






