import pandas as pd 
import numpy as np

def Read_file(path):
    Data=pd.read_csv(path)
    print(Data.head(5))
    return Data 

def Data_types(Data):
    print(Data.dtypes)
    print("\n")
    print(Data.info())

def main():
    Data=Read_file("Student_performance_data _.csv")
    Data_types(Data)
if __name__ == "__main__":
    main()