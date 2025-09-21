🚢 Titanic Survival Prediction – Logistic Regression Case Study

✅ Author: Yashshree Ganesh Kalokhe

---

📌 Project Overview

This project applies Logistic Regression to predict survival outcomes on the Titanic dataset. It includes steps like:

* Data cleaning and preprocessing
* Feature encoding and scaling
* Model training and evaluation
* Correlation visualization

The purpose of this case study is to demonstrate a solid understanding of machine learning workflows, data handling, and performance evaluation in Python.

---

## 🧠 Dataset Description

The dataset used is a version of the classic **Titanic dataset**, which includes information about passengers like age, gender, ticket class, and whether they survived.

**Target Variable:**

* `Survived` -> 1
* `Not survived` -> 0

---

## 🗂️ Project Structure

```
titanic-logistic-regression/
│
├── titanic.csv                 # Input dataset
├── titanic_logistic.py         # Main Python script (your code)
├── README.md                   # Documentation (this file)
```

---

## 🛠️ Dependencies

* Python 3.x
* pandas
* numpy
* seaborn
* matplotlib
* scikit-learn

### 📦 Install Requirements

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## ▶️ How to Run

1. Make sure `titanic.csv` is in the same folder as the script.
2. Run the Python script:

```bash
python titanic_logistic.py
```

---

## 🔍 Functional Breakdown

Here’s a step-by-step explanation of what the script does:

---

### 1. `TitanicLogistic(datapath)`

This is the main function that performs the entire pipeline.

---

#### ✅ Step 1: Load Dataset

```python
df = pd.read_csv(datapath)
```

* Loads the Titanic dataset from a CSV file.
* Displays the first few rows and dataset dimensions.

---

#### ✅ Step 2: Encode Categorical Features

```python
df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
```

* Encodes the `Sex` column to numeric values:

  * Male → 0
  * Female → 1

---

#### ✅ Step 3: Drop Unnecessary Columns

```python
df.drop(columns=['PassengerId', 'zero', 'Cabin'], inplace=True)
```

* Removes irrelevant or redundant columns to reduce noise.

---

#### ✅ Step 4: Handle Missing Values

```python
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

* Fills missing values in the `Embarked` column using the mode (most frequent value).

---

#### ✅ Step 5: Correlation Heatmap

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

* Visualizes feature correlation using Seaborn.
* Helps understand relationships between variables and the target (`Survived`).

> 🛠️ **Note:** There’s a bug in the code here:
> You used `df.corr` instead of `df.corr()`. The correct usage includes parentheses.

---

#### ✅ Step 6: Feature & Target Separation

```python
x = df.drop(columns=['Survived'])
y = df['Survived']
```

* `x` → Feature matrix
* `y` → Target labels

---

#### ✅ Step 7: Feature Scaling

```python
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
```

* Standardizes the feature matrix to have 0 mean and unit variance.
* Essential for Logistic Regression and other models that are sensitive to feature scales.

> ⚠️ **Note:** There’s a typo here:
> Use `fit_transform()` instead of `fit_tranform()`.

---

#### ✅ Step 8: Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.2, random_state=42)
```

* Splits data into 80% training and 20% testing sets.

---

#### ✅ Step 9: Model Training & Prediction

```python
model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
```

* Trains a Logistic Regression model and generates predictions.

---

#### ✅ Step 10: Model Evaluation

```python
accuracy_score(y_test, y_predict)
confusion_matrix(y_test, y_predict)
```

* Prints the **accuracy score**.
* Displays the **confusion matrix** to understand prediction performance in more detail.

---

## 🧪 Sample Output

```
Dataset loaded successfully 
   PassengerId  Pclass     Name   Sex   Age  ...  Embarked  Survived
0            1       3  Braund   Male  22.0  ...     S         0
...

Dimensions of dataset is  (891, 12)

Encoded dataset:
   Pclass  Sex   Age  SibSp  Parch  Fare  Embarked  Survived
0       3    0  22.0      1      0  7.25         S         0
...

[Correlation heatmap displayed]

Accuracy of the model is  0.81

The confusion matrix is:
[[90  9]
 [15 45]]
```

---

## 💼 Why This Project Matters

This project showcases my ability to:

* Clean and transform messy real-world data
* Apply and evaluate a basic machine learning model
* Visualize data to support feature selection
* Build a reproducible and modular pipeline


---

## 📧 Contact

* **Email:** kalokheyashshree@gmail.com
* **GitHub:** https://github.com/Yash1723vk/Projects/new/main/Titanic

---
