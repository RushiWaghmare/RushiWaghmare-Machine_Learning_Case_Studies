from sklearn import tree

def MarvellousClassifier():
    # Feature encoding
    Features = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 1], [35, 1], [92, 0], [35, 1], [35, 1], [35, 1], [35, 1]]

    # Label Encoding
    Labels = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1]


    # Decide the algorithm
    obj = tree.DecisionTreeClassifier()

    # Train the algorithm
    obj =obj.fit(Features,Labels)

    #Test the model
    ret=obj.predict([[96, 0]])
    if ret ==1:
        print("your ball looks like tennis ball")
    else:
        print("your ball looks like Cricket ball")

def main():
    print("-------------Ball type classification case study----------------")

    MarvellousClassifier()





  


if __name__ == "__main__":
    main()
