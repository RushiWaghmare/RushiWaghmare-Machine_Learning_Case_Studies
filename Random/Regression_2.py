import pandas as pd
import numpy as np

def Read_Data(Path):
    Data = pd.read_csv(Path)
    print(Data.head(5))
    print(Data.info())
    return Data

def Data_initialise(Data):
    X = Data["X"].values
    Y = Data["Y"].fillna(Data["Y"].mean()).values  # Convert to array
    return X, Y

def Mean_X_Y(X, Y):
    X_bar = np.mean(X)
    Y_bar = np.mean(Y)
    N = len(X)
    return X_bar, Y_bar, N

def Deviation(X_bar, Y_bar, N, X, Y):
    Sum_dev = np.sum((X - X_bar) * (Y - Y_bar))
    Deno = np.sum((X - X_bar) ** 2)
    print(f"Sum of Product of Deviation of X and Y: {Sum_dev}")
    print(f"Sum of Square of Deviation of X: {Deno}")
    return Sum_dev, Deno

def Slope(Sum_dev, Deno):
    M = Sum_dev / Deno
    print("Slope of Line is", M)
    return M

def Y_intercept(M, X_bar, Y_bar):
    C = Y_bar - (M * X_bar)
    print(f"Value of C (Y Intercept): {C}")
    return C

def PredictionOf_Y(M, C, X):
    Y_pred = M * X + C
    return Y_pred

def R_Square(Y, Y_pred):
    Numerator = np.sum((Y - Y_pred) ** 2)
    Denominator = np.sum((Y - np.mean(Y)) ** 2)
    R_sq = 1 - (Numerator / Denominator)
    print(f"Value of R_Square is: {R_sq}")
    return R_sq

def main():
    Data = Read_Data("train.csv")

    X, Y = Data_initialise(Data)

    X_bar, Y_bar, N = Mean_X_Y(X, Y)

    Sum_dev, Deno = Deviation(X_bar, Y_bar, N, X, Y)

    M = Slope(Sum_dev, Deno)

    C = Y_intercept(M, X_bar, Y_bar)

    Y_pred = PredictionOf_Y(M, C, X)

    R_Square(Y, Y_pred)

if __name__ == "__main__":
    main()
