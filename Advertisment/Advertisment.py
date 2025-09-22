from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

def Advertise(datapath):
   df = pd.read_csv(datapath)
   print("Dataset sample is")
   print(df.head())

   print("Clean the dataset")
   df.drop(columns = ['Unnamed: 0'], inplace = True)

   print("Updated dataset ")
   print(df.head())

   print("Missing values in each columns" , df.isnull().sum())

   print("Statistical data is ")
   print(df.describe())

   print("Correlation matri")
   print(df.corr())

   plt.figure(figsize= (10, 5))
   sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm')
   plt.title("Advertisment Coorelation HeatMap")
   plt.show()

   sns.pairplot(df)
   plt.suptitle("Pairplot of features", y = 1.02)
   plt.show()

   x = df[['TV', 'Radio', 'Newspaper']]
   y = df[['Sales']]

   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)
   model = LinearRegression()
   model.fit(x_train, y_train)

   y_pred = model.predict(x_test)

   mse = metrics.mean_squared_error(y_test,y_pred)
   rmse = np.sqrt(mse)
   r2 = metrics.r2_score(y_test, y_pred)

   print("Mean squared error is ", mse)
   print("Root Mean squared error is ", rmse)
   print("R square ", r2)

   print("Model coefficient are ")
   for col, coef in zip(x.columns, model.coef_):
      print(f"{col} : {coef}")

   print("Y intercept is ", model.intercept_)   

   plt.figure(figsize= (8, 5))
   plt.scatter(y_test, y_pred , color = 'blue')
   plt.xlabel("Actual Sales")
   plt.ylabel("Predicted sales")
   plt.title("Advertisment")
   plt.grid(True)
   plt.show()


def main():
  Advertise("Advertising.csv")

if __name__ == "__main__":
    main()    
