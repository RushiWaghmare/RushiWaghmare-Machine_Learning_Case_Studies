import pandas as pd
from sklearn.preprocessing import LabelEncoder
def Read_file(Path):    
    Data=pd.read_csv(Path)
    #print(Data.head())
    print(Data.info())
    return Data

def Data_Maipulation(Data):
    label_encoder = LabelEncoder()
    Data["BusinessTravel"]= label_encoder.fit_transform(Data["BusinessTravel"])
    print(Data["BusinessTravel"].head(5))


def main():
    Data=Read_file("Employe.csv")
    Data_Maipulation(Data)
if __name__=="__main__":
    main()