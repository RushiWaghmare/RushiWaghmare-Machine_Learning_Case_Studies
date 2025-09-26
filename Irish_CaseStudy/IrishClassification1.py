from sklearn import tree
from sklearn.datasets import load_iris

def main():
    print("--------Iris flower Classification case study----------")

    iris = load_iris()
    #print(shape(iris))   #shape gives dimentions of data
    #print(type(iris))
    
    Fetures= iris.data
    Lables = iris.target

    print("Fetures are: ",Fetures)
    print("Lables are: ",Lables)

if __name__ == "__main__":
    main()