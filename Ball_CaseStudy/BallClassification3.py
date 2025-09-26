from sklearn import tree

# Rough =1
# Smooth =0

# Tennis = 1
# Cricket = 2
def main():
    print("Ball Classification case study")
    
    # Feature encoding
    Features = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 1], [35, 1], [92, 0], [35, 1], [35, 1], [35, 1], [35, 1]]

    # Label Encoding
    Labels = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1]


    # Decide the algorithm
    obj = tree.DecisionTreeClassifier()

    # Train the algorithm
    obj =obj.fit(Features,Labels)

    #Test the model

    print(obj.predict([[96,0]]))
    '''
    prediction = clf.predict([[96, 0]])
    print(f"The predicted class for a ball with size 96 and smooth texture is: {prediction[0]}")
'''

if __name__ == "__main__":
    main()
