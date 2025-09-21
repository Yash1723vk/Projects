# 🎗️ Breast Cancer Detection – Support Vector Machine (SVM)

### ✅ Author: Yashshree Ganesh Kalokhe
---

## 📌 Project Overview

This project demonstrates a supervised machine learning approach for detecting **malignant vs benign breast cancer tumors** using the **Support Vector Machine (SVM)** algorithm with a linear kernel.

The model is trained on the **Breast Cancer Wisconsin Diagnostic Dataset**, a popular dataset used in medical ML research. This project includes:

* Data loading and preprocessing
* Feature scaling
* Model training with SVM
* Accuracy and confusion matrix evaluation

---

## 🧠 Dataset Description

The dataset is loaded using:

```python
from sklearn.datasets import load_breast_cancer
```

It includes **30 numerical features** computed from digitized images of fine needle aspirate (FNA) of breast masses.

### 🗂️ Features:

Some of the features include:

* Mean radius
* Mean texture
* Mean perimeter
* Mean smoothness
* And 26 more...

### 🎯 Target:

* **0**: Malignant
* **1**: Benign

---

## 🗂️ Project Structure

```
breast-cancer-svm/
│
├── breast_cancer_svm.py          # Python script (your code)
├── README.md                     # Documentation (this file)
```

---

## 📦 Dependencies

* Python 3.x
* pandas
* numpy
* scikit-learn

### 📦 Install Dependencies

```bash
pip install pandas numpy scikit-learn
```

---

## ▶️ How to Run

Just run the main script:

```bash
python breast_cancer_svm.py
```

---

## 🔍 Functional Breakdown

### 🔬 1. Load Dataset

```python
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
```

* Loads 569 samples with 30 features each.
* Creates a DataFrame for inspection and adds the target column.

---

### 🧼 2. Feature Scaling

```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
```

* Standardizes the dataset (zero mean and unit variance).
* SVM is sensitive to feature scales, so this step is crucial.

---

### 🔀 3. Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
```

* 80% training, 20% testing split.
* `random_state=42` ensures reproducibility.

---

### 🧠 4. Train SVM Classifier

```python
model = SVC(kernel='linear', C=1)
model.fit(x_train, y_train)
```

* Uses a **linear kernel**, suitable for linearly separable data.
* `C=1` is the regularization parameter (controls margin size).

---

### 📈 5. Model Evaluation

```python
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
```

* **Accuracy:** Proportion of correct predictions.
* **Confusion Matrix:** Breakdown of TP, FP, TN, FN.

---

## 📊 Sample Output

```bash
(569, 31)
classes:  {'malignant': 0, 'benign': 1}
accuracy is:  0.9736842105263158

confusion matrix:
[[43  0]
 [ 3 68]]
```

* This shows a **97.3% accuracy**, which is excellent for a baseline linear SVM.
* Low false positives/negatives — very promising!

---

## 🧪 Potential Improvements

* Try different kernels (`'rbf'`, `'poly'`) and compare performance.
* Use **cross-validation** for better generalization.
* Plot the **ROC curve** and compute **AUC score**.
* Try **feature selection** to reduce dimensionality.
* Use **GridSearchCV** to tune `C` and `kernel` parameters.

---

## 🧠 Key Skills Demonstrated

This project reflects proficiency in:

* Applying **supervised learning algorithms**
* Using **SVM for classification**
* Preprocessing with **StandardScaler**
* Model evaluation using **metrics and confusion matrix**
* Real-world medical data handling

---

## 📧 Contact

* Email: kalokheyashshree@gmail.com
* GitHub: https://github.com/Yash1723vk/Breast_Cancer
