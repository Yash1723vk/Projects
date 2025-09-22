# 🧠 Head Size vs. Brain Weight Prediction – Linear Regression

### ✅ Author: Yashshree Ganesh Kalokhe

---

## 📌 Project Overview

This project explores the relationship between **head size** and **brain weight** using a **simple linear regression model**. The model attempts to predict brain weight from head size, highlighting how even basic physiological metrics can exhibit meaningful statistical correlations.

This case study involves:

* Dataset loading and inspection
* Exploratory Data Analysis (EDA)
* Building a regression model
* Evaluating performance with **MSE, RMSE, and R²**
* Plotting the regression line over actual data points

This is a classic **univariate regression** problem, ideal for demonstrating foundational modeling skills in a real-world biological dataset.

---

## 🧠 Dataset Description

The dataset (`headbrain.csv`) contains data for a group of individuals with the following attributes:

### 🗂️ Columns:

| Column Name           | Description                      |
| --------------------- | -------------------------------- |
| `Head Size(cm^3)`     | Head volume in cubic centimeters |
| `Brain Weight(grams)` | Weight of the brain in grams     |

---

## 🗂️ Project Structure

```
head-brain-regression/
│
├── HeadBrain.py      # Python script (your code)
├── headbrain.csv     # Dataset file
├── README_HEAD.md    # Documentation (this file)
```

---

## 📦 Dependencies

* Python 3.x
* pandas
* numpy
* matplotlib
* scikit-learn

### 📦 Install with pip:

```bash
pip install pandas numpy matplotlib scikit-learn
```
---

## 🔍 Functional Breakdown

### 📥 1. Load and Explore the Dataset

```python
df = pd.read_csv("headbrain.csv")
df.head()
df.describe()
```

* Reads the dataset using `pandas`
* Displays the first few rows and statistical summary

---

### 🧠 2. Define Variables

```python
x = df[['Head Size(cm^3)']]
y = df[['Brain Weight(grams)']]
```

* `x`: Feature (independent variable)
* `y`: Target (dependent variable)

---

### 🔀 3. Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

* 80% data used for training
* 20% held out for testing

---

### 📈 4. Train Linear Regression Model

```python
model = LinearRegression()
model.fit(x_train, y_train)
```

* A **Simple Linear Regression** model is fit to the training data

---

### 🧪 5. Model Evaluation

```python
mse = mean_squared_error(y_test, y_prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_prediction)
```

* **MSE** (Mean Squared Error): How far off predictions are
* **RMSE**: Root of MSE, in same units as target
* **R² Score**: Proportion of variance explained

---

### 🖼️ 6. Visualization

```python
plt.scatter(x_test, y_test)
plt.plot(x_test, y_prediction)
```

* **Scatter plot** of actual data
* **Regression line** showing model prediction

---

## 📊 Sample Output

```
Mean Squared Error is  2343.53
Slope of line  [0.2633]
Intercept  [325.57]
Root Mean Squared error  48.41
R squared value  0.64
```

This indicates:

* A moderately strong linear relationship (R² ≈ 0.64)
* The brain weight increases by \~0.263 grams for every 1 cm³ increase in head size.

---

## 🎯 Key Skills Demonstrated

This project showcases:

* 🧮 Mastery of **Linear Regression Modeling**
* 📊 Strong **EDA & Visualization** using `matplotlib`
* 📐 Understanding of **model interpretability** (slope & intercept)
* 📏 Regression model **performance metrics**
* 💻 Data pipeline creation from loading to model evaluation

---

## ✅ Why This Project Matters

* This project mimics real-world applications in **neuroscience**, **medicine**, and **biometry**
* Demonstrates how **basic statistical learning** methods can uncover meaningful biological trends
* Reinforces key machine learning concepts like **model training**, **testing**, and **evaluation**

---

## 📧 Contact

* Email: [kalokheyashshree@gmail.com](mailto:kalokheyashshree@gmail.com)
* GitHub: [https://github.com/Yash1723vk](https://github.com/Yash1723vk)
