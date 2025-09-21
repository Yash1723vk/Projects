# 🍷 Wine Classification – K-Nearest Neighbors (KNN) Model

### ✅ Author: Yashshree Ganesh Kalokhe

---

## 📌 Project Overview

This project demonstrates a machine learning solution for **predicting wine classes** using the **K-Nearest Neighbors (KNN)** algorithm. It includes data preprocessing, hyperparameter tuning (for optimal `k`), and performance evaluation with a visualization of accuracy scores across different `k` values.

The entire pipeline is implemented in Python using `pandas`, `scikit-learn`, and `matplotlib`.

---

## 🧠 Dataset Description

The dataset used is a version of the **Wine Dataset**, commonly used in classification problems.

**Target Variable:**

* `Class`: Categorical label representing the wine type (e.g., Class 1, 2, or 3)

**Features:**

* Multiple numerical attributes of wine like alcohol content, color intensity, hue, etc.

---

## 🗂️ Project Structure

```
wine-knn-classifier/
│
├── WinePredictor.csv           # Input dataset file
├── wine_knn_classifier.py      # Python script (your code)
├── README.md                   # Project documentation (this file)
```

---

## 🛠️ Dependencies

* Python 3.x
* pandas
* numpy
* matplotlib
* scikit-learn

### 📦 Install Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```
---

## 🔍 Functional Breakdown

The core function of this script is `WinePredictor(datapath)`, which handles the full ML workflow.

---

### ✅ Step-by-Step Breakdown

---

### 1. Load and Clean Data

```python
df = pd.read_csv(datapath)
df.dropna(columns=['Class'])
```

* Loads the CSV dataset into a DataFrame.
---

### 2. Feature and Label Separation

```python
x = df.drop(columns=['Class'])
y = df['Class']
```

* `x`: Feature matrix
* `y`: Target labels

---

### 3. Feature Scaling

```python
scaler = StandardScaler
x_scale = scaler.fit_transform(x)
```

```python
scaler = StandardScaler()
x_scale = scaler.fit_transform(x)
```

* Standardizes the feature values to improve KNN performance.

---

### 4. Train-Test Split

```python
train_test_split(x_scale, y, test_size=0.2, random_state=42)
```

* Splits the dataset into training and testing sets (80/20).

---

### 5. Hyperparameter Tuning (Optimal k)

```python
for k in range(1, 21):
    ...
```

* Trains 20 KNN models with `k = 1` to `20`.
* Stores accuracy scores for each value of `k`.

---

### 6. Accuracy Plot

```python
plt.plot(k_range, accuracy_scores)
```

* Visualizes the accuracy score vs. `k` value.
* Helps select the best `k` for model performance.
---

### 7. Final Model Training with Best k

```python
best_k = k_range[accuracy_scores.index(max(accuracy_scores))]
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(...)
```

* Selects the best `k` with highest accuracy.
* Retrains the final model using optimal `k`.

---

### 8. Final Evaluation

```python
accuracy_score(y_test, y_predict)
```

* Outputs the best model's accuracy on the test set.

---

## 📊 Sample Output

```
Best value of k is  5
Best Accuracy is  97.22
[Accuracy vs K plot displayed]
```

---

## 💼 Why This Project Matters

This project demonstrates my ability to:

* Work with real-world classification problems
* Preprocess and standardize datasets correctly
* Tune hyperparameters using a systematic and visual approach
* Train and evaluate ML models using industry-standard tools

---

## 📧 Contact

* Email: kalokheyashshree@gmail.com
* GitHub: https://github.com/Yash1723vk/Projects/new/main/WinePredictor

---
