import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def loadData(file_path):
    df = pd.read_csv(file_path)
    print("Dataset gets loaded in computer successfully")
    return df


def getInformation(df):
    print("Information about the loaded dataset is ")
    print("Shape of dataset ", df.shape)
    print("Columns ", df.columns)
    print("Missing values ", df.isnull().sum())

def encodeData(df):
    df["variety"] = df ["variety"].map({"Setosa" : 0, "Versicolor" : 1, "Virginica" : 2})
    return df

def splitFeatureLable(df):
    X = df.drop("variety" , axis = 1)
    Y = df["variety"]
    return X, Y

def split(X,Y,size = 0.2):
    return train_test_split(X,Y, test_size = size)

def main():
    data = loadData("iris.csv")
    print(data.head())

    getInformation(data)

    print("Data after encoding ")
    data = encodeData(data)
    print(data.head())

    print("Spliting features and lables ")
    Independent , Dependent = splitFeatureLable(data)
    print(Independent.head())
    print(Dependent.head())

    X_train, X_test, Y_train, Y_test = split(Independent, Dependent, 0.3)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_train.shape)



if __name__ == "__main__":
    main()    
