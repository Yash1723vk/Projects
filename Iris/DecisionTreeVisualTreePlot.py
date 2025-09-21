from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def DecisionTree():
    iris = load_iris() #data set come

    X = iris.data   # independent
    Y = iris.target #dependent

    X_train , X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2) #split the data
    
    model = DecisionTreeClassifier() #take default values

    model = model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_predict)

    print("Accuracy is ", accuracy*100)

    plt.figure(figsize = (12, 10))
    plot_tree(model, filled = True, feature_names = iris.feature_names, class_names = iris.target_names)
    print("Decision tree classifier")
    plt.show()

def main():
   DecisionTree()

if __name__ == "__main__":
    main()    

# why acc is diff is due to random state    