import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def Data_read(path):
    Data = pd.read_csv(path)
    print(Data.head())
    return Data

def Data_Initialise(Data):
    label_encoder = LabelEncoder()

    Data["Location"] = label_encoder.fit_transform(Data["Location"])
    Data["Grade"] = label_encoder.fit_transform(Data["Grade"])
    Data["Gender"] = label_encoder.fit_transform(Data["Gender"])

    X = Data[["Gender", "Grade", "Location", "Size (cm)", "Patient_Age"]]
    Y = Data["Tumor_Type"]

    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, Y

def Data_Train(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def DecisionTree_Classifier(X_train, X_test, Y_train, Y_test):
    DT = DecisionTreeClassifier()
    DT.fit(X_train, Y_train)
    Y_pred = DT.predict(X_test)
    print("Decision Tree Testing Accuracy:", accuracy_score(Y_test, Y_pred) * 100, "%")

def Logistic_Reggression(X_train, X_test, Y_train, Y_test):
    LR = LogisticRegression(max_iter=1000)
    LR.fit(X_train, Y_train)
    Y_pred = LR.predict(X_test)
    print("Logistic Regression Testing Accuracy:", accuracy_score(Y_test, Y_pred) * 100, "%")

def RandomForest_Classifier(X_train, X_test, Y_train, Y_test):
    # Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    RF = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, Y_train)

    best_rf = grid_search.best_estimator_
    Y_pred = best_rf.predict(X_test)
    print("Random Forest Testing Accuracy:", accuracy_score(Y_test, Y_pred) * 100, "%")
    print("Best Hyperparameters:", grid_search.best_params_)

def main():
    Data = Data_read("brain_tumor_dataset.csv")
    X, Y = Data_Initialise(Data)
    X_train, X_test, Y_train, Y_test = Data_Train(X, Y)
    # DecisionTree_Classifier(X_train, X_test, Y_train, Y_test)
    # Logistic_Reggression(X_train, X_test, Y_train, Y_test)
    RandomForest_Classifier(X_train, X_test, Y_train, Y_test)

if __name__ == "__main__":
    main()
