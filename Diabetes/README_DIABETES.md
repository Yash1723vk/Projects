# 🩺 Diabetes Prediction using Logistic Regression

This project uses **Logistic Regression** to predict whether a person is diabetic based on medical diagnostic measurements. The dataset used is the popular **Pima Indians Diabetes Dataset**.

---

## 📁 Project Structure

```
.
├── diabetes.csv         # Dataset file
├── Diabetes.py  # Main Python script
└── README_DIABETES.md            # Project description
```

---

## 📌 Requirements

Ensure you have Python 3.x installed. You can install required libraries using:

```bash
pip install pandas numpy scikit-learn
```

---

## 🧠 What the Script Does

* Loads the **diabetes.csv** dataset using `pandas`.
* Splits the dataset into **features (X)** and **labels (Y)**.
* Performs an **80/20 train-test split** using `train_test_split`.
* Trains a **Logistic Regression** model on the training set.
* Prints:

  * The dataset's columns, head, and shape
  * The shape of `X` and `Y`
  * The **training accuracy** of the model

---

## 📊 Dataset Information

The dataset contains medical data including:

* `Pregnancies`
* `Glucose`
* `BloodPressure`
* `SkinThickness`
* `Insulin`
* `BMI`
* `DiabetesPedigreeFunction`
* `Age`
* `Outcome` (Target: 1 = Diabetic, 0 = Not Diabetic)

---

## ✅ Example Output

```
Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')
   Pregnancies  Glucose  BloodPressure  ...  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72  ...                     0.627   50        1
...
(768, 9)
(768, 8)
(768,)
Training Accuracy 
0.78
```

---

📧 Contact
Email: kalokheyashshree@gmail.com
GitHub: https://github.com/Yash1723vk/Diabetes
