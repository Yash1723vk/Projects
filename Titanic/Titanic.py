import pandas as pd
import numpy as np

from matplotlib.pyplot import figure , show
import seaborn as sns
from seaborn import countplot
import matplotlib as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score , confusion_matrix

def TitanicLogistic(datapath):
    df = pd.read_csv(datapath)
    print("Dataset loaded sucessfully ", df.head())

    print("Dimensions of dataset is ", df.shape)

    df['Sex'] = df['Sex'].map({'Male' : 0, 'Female' : 1})
    print("The dataset is ", df.head())
     
    df.drop(columns = ['PassengerId' , 'zero', 'Cabin'],  inplace= True) 

    print("Dimensions of dataset is ", df.shape)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

    plt.figure(figsize = (10,6))
    sns.heatmap(df.corr(), annot  =True , cmap= 'coolwarm')
    plt.title("Correlation heat map")
    show()

    x = df.drop(columns= ['Survived'])
    y = df['Survived']

    print("Dimensions of target ", x.shape)
    print("Dimensions of labels ", y.shape)

    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)
    
    x_train ,x_test ,y_train, y_test = train_test_split(x_scale ,y, test_size= 0.2, random_state= 42)

    model = LogisticRegression()

    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    Accuracy = accuracy_score(y_test , y_predict)
    print("Accuracy of the model is ", Accuracy)

    cm = confusion_matrix(y_test , y_predict)
    print("The confusion is ", cm)

def main():
    TitanicLogistic("titanic.csv")

if __name__ == "__main__":
    main()   
