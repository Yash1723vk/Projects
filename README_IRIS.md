🌸 Iris Dataset Case Study
✅ Author: Yashshree Ganesh Kalokhe
🏢 Company Context: Scale AI (Prospective Candidate)


📌 Project Overview

This project demonstrates a the famous Iris dataset. The goal is to prepare the dataset for machine learning by loading, exploring, encoding, and splitting the data for model training.

This pipeline is implemented in Python using industry-standard libraries such as pandas and scikit-learn. Each function in the script is modular, reusable, and aligns with 
real-world data science practices.

🧠 Dataset Description

The Iris dataset consists of 150 samples of iris flowers, with 4 numerical features (sepal length, sepal width, petal length, petal width) and a
target column: the variety of Iris plant.

Target Classes:
* Setosa (0)
* Versicolor (1)
* Virginica (2)

---

🗂️ Folder Structure

iris-case-study/
│
├── iris.csv                 # Input dataset file
├── iris_preprocessing.py    # Main script (your provided code)
├── README.md                # Project documentation (this file)

---

📜 How to Use

1. Clone the repository or download the files.
2. Make sure `iris.csv` is in the same directory as the script.
3. Run the script using Python 3

---

📦 Dependencies

* Python 3.x
* pandas
* scikit-learn

Install them using:

pip install pandas scikit-learn

---

🔍 Functional Breakdown

Below is a detailed explanation of all the components in the script.

---

1. loadData(file_path)

```python
def loadData(file_path):
    df = pd.read_csv(file_path)
    print("Dataset gets loaded in computer successfully")
    return df
```

* Purpose: Loads the Iris dataset from a `.csv` file using pandas.
* Returns: A DataFrame object containing the dataset.
* Message: Confirms successful data loading.

---

2. getInformation(df)

```python
def getInformation(df):
    print("Information about the loaded dataset is ")
    print("Shape of dataset ", df.shape)
    print("Columns ", df.columns)
    print("Missing values ", df.isnull().sum())
```

* Purpose: Displays basic info about the dataset.
* Outputs:

  * Shape of the dataset (rows, columns)
  * Column names
  * Count of missing values per column

---

3. encodeData(df)

```python
def encodeData(df):
    df["variety"] = df["variety"].map({"Setosa": 0, "Versicolor": 1, "Virginica": 2})
    return df
```

* Purpose: Encodes the categorical target variable `variety` into numerical labels.
* Mapping:

  * `Setosa` → 0
  * `Versicolor` → 1
  * `Virginica` → 2
* Why?
*  Machine learning models require numerical input.

---

4. splitFeatureLable(df)

```python
def splitFeatureLable(df):
    X = df.drop("variety", axis=1)
    Y = df["variety"]
    return X, Y
```

* Purpose: Separates the dataset into:

  * `X`: Feature matrix (input variables)
  * `Y`: Target vector (output labels)
* **Output:** Two separate DataFrames for features and labels.

---

5. split(X, Y, size=0.2)

```python
def split(X, Y, size=0.2):
    return train_test_split(X, Y, test_size=size)
```

* Purpose: Splits data into training and testing subsets.
* Parameters:

  * `X`: Features
  * `Y`: Labels
  * `size`: Proportion of test set (default = 20%)
  * 
* Returns:

  * `X_train`, `X_test`, `Y_train`, `Y_test`

---

6. main()

```python
def main():
    ...
```

* Driver function that calls all the above functions in sequence:

  1. Loads the data
  2. Displays basic info
  3. Encodes the target column
  4. Splits the dataset into features and labels
  5. Splits features and labels into training and testing sets
  6. Displays dimensions of training and testing data

---

🧪 Sample Output

```
Dataset gets loaded in computer successfully
   sepal_length  sepal_width  petal_length  petal_width    variety
0           5.1          3.5           1.4          0.2     Setosa
...

Information about the loaded dataset is 
Shape of dataset  (150, 5)
Columns  Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'variety'], dtype='object')
Missing values  sepal_length    0
...

Data after encoding 
   sepal_length  sepal_width  petal_length  petal_width  variety
0           5.1          3.5           1.4          0.2        0
...

Spliting features and lables 
   sepal_length  sepal_width  petal_length  petal_width
0           5.1          3.5           1.4          0.2
...

(105, 4)
(45, 4)
(105,)
(105,)
```

---

📧 Contact

* Email: kalokheyashshree@gmail.com
* GitHub: https://github.com/Yash1723vk/

---

Let me know if you'd like to add **visuals**, a **license**, or a **requirements.txt** for even more professionalism.
