from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def HeadBrainLinear(datapath):
    Line = "*" * 50
    df = pd.read_csv(datapath)

    print(Line)
    print("First few records of the dataset are ")
    print(Line)
    print(df.head())
    print(Line)

    print(Line)
    print("Statistical information of the dataset")
    print(Line)
    print(df.describe())
    print(Line)

    x = df[['Head Size(cm^3)']] #as it is df you know df ds and dp
    y = df[['Brain Weight(grams)']]

    print("Independent variables are Head Size")
    print("Dependent variables are Brain Weight")

    print("Total records in dataset " , x.shape) #shape is a property

    x_train , x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state= 42)

    print("Dimensions of training" , x_train.shape)
    print("Dimensions of training" , x_test.shape)

    model = LinearRegression()

    model.fit(x_train, y_train)
    y_prediction = model.predict(x_test)

    mse = mean_squared_error(y_test, y_prediction)
    rmse = np.sqrt(mse) 
    r2 = r2_score(y_test, y_prediction)

    print("Visual represtation")

    plt.figure(figsize= (8,5))
    plt.scatter(x_test, y_test, color = 'blue', label = 'actual')
    plt.plot(x_test.values.flatten(), y_prediction, color = 'red', linewidth = 2, label = "Regression")
    plt.xlabel("Head Size(cm^3)")
    plt.ylabel("Brain weight(grams)")
    plt.title("Head Brain Regression")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("Mean Squared Error is ", mse)
    print("Slope of line ", model.coef_[0])
    print("Intercept ", model.intercept_)
    print("Root Mean Squared error ", rmse)
    print("R squared value ", r2) #is like accuracy

    print("Result of case study")
def main():
  HeadBrainLinear("headbrain.csv")

if __name__ == "__main__":
    main()
