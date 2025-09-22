# 📈 Advertising Sales Prediction – Linear Regression

### ✅ Author: Yashshree Ganesh Kalokhe

---

## 📌 Project Overview

This project demonstrates a **supervised machine learning regression approach** to **predict product sales** based on different advertising channel spends using the **Linear Regression** algorithm.

The dataset contains advertising budget allocations across **TV, Radio, and Newspaper**, and the corresponding **sales figures**. The project showcases:

* Data cleaning and exploration
* Correlation analysis with heatmaps and pairplots
* Model building using **Linear Regression**
* Evaluation using **MSE, RMSE, R² Score**
* Visualization of predictions vs actual values

This project reflects real-world skills in data science and regression modeling — essential for decision-making in marketing analytics.

---

## 🧠 Dataset Description

The dataset is sourced from a well-known marketing study and is stored in a CSV file named **`Advertising.csv`**.

### 🗂️ Features:

| Column      | Description                                 |
| ----------- | ------------------------------------------- |
| `TV`        | Advertising budget spent on TV (in \$1000s) |
| `Radio`     | Advertising budget spent on Radio           |
| `Newspaper` | Advertising budget spent on Newspaper       |
| `Sales`     | Actual product sales (in \$1000s)           |

---

## 🗂️ Project Structure

```
advertising-regression/
│
├── Advertisment.py     # Main Python script
├── Advertising.csv     # Dataset file
├── README_AD.md        # Documentation (this file)
```

---

## 📦 Dependencies

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

### 💻 Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---
## 🔍 Functional Breakdown

### 📥 1. Load and Clean the Dataset

```python
df = pd.read_csv("Advertising.csv")
df.drop(columns=['Unnamed: 0'], inplace=True)
```

* Loads the dataset using `pandas`
* Drops irrelevant index column

### 📊 2. Data Inspection & EDA

```python
df.describe()
df.isnull().sum()
df.corr()
sns.heatmap(...)
sns.pairplot(...)
```

* Prints summary statistics and missing value checks
* Visualizes feature correlation using **heatmaps**
* Uses **pairplot** to inspect bivariate relationships

### 🧠 3. Model Building – Linear Regression

```python
x = df[['TV', 'Radio', 'Newspaper']]
y = df[['Sales']]
model = LinearRegression()
model.fit(x_train, y_train)
```

* Features and target variable are defined
* Train/test split with `train_test_split` (80/20)
* Model is trained on training data

### 📈 4. Model Evaluation

```python
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

* Evaluates model with:

  * **MSE (Mean Squared Error)**
  * **RMSE (Root Mean Squared Error)**
  * **R² Score** – measure of model's predictive power

### 📉 5. Model Insights and Visualization

```python
model.coef_
model.intercept_
plt.scatter(y_test, y_pred)
```

* Prints model coefficients for interpretability
* Visualizes **actual vs predicted sales** in a scatter plot

---

## 📊 Sample Output

```
Mean squared error is  1.91
Root Mean squared error is  1.38
R square  0.89

Model coefficient are 
TV : [0.0446]
Radio : [0.189]
Newspaper : [-0.001]
Y intercept is  [2.93]
```

* The model performs with high accuracy (R² ≈ 0.89), meaning \~89% of the sales variation is explained by advertising budgets.

---

## 🧠 Key Skills Demonstrated

This project demonstrates strong capabilities in:

* 📊 **Exploratory Data Analysis** (EDA)
* 🧮 **Linear Regression Modeling**
* 📉 **Error Metrics & Model Evaluation**
* 🖼️ **Visualization with Matplotlib & Seaborn**
* 📦 **Handling real-world datasets**
* 🤖 **Applying predictive analytics to business cases**

---

## ✅ Why This Project Matters

Predictive modeling for marketing spend is a **core use case in applied data science**. This project mirrors real-world work where businesses must:

* Allocate advertising budgets efficiently
* Forecast sales based on marketing strategy
* Justify ad spending using interpretable models

---

## 📧 Contact

* Email: [kalokheyashshree@gmail.com](mailto:kalokheyashshree@gmail.com)
* GitHub: [https://github.com/Yash1723vk](https://github.com/Yash1723vk)

---
