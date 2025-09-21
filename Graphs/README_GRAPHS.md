🌼 Iris Dataset Visualizations – Exploratory Data Analysis (EDA)

### ✅ Author: Yashshree Ganesh Kalokhe
---

## 📌 Project Overview

This project demonstrates various **exploratory data analysis (EDA)** and **data visualization** techniques using the classic **Iris dataset**. The goal is to gain insights into the relationships between different flower measurements and species types using:

* Histogram
* Boxplot
* Pairplot
* 3D scatter plot

These visualizations help in understanding data distributions, feature correlations, and class separability — essential steps before any machine learning modeling.

---

## 🧠 Dataset Description

The **Iris dataset** consists of **150 samples** of iris flowers with the following features:

* **Sepal Length**
* **Sepal Width**
* **Petal Length**
* **Petal Width**
* **Variety** (target class: Setosa, Versicolor, Virginica)

---

## 🗂️ Project Structure

```
iris-visualizations/
│
├── iris.csv                        # Dataset (if loaded manually)
├── histogram_plot.py               # Histogram of sepal length
├── boxplot_petal_length.py         # Boxplot of petal length by variety
├── pairplot_iris.py                # Pairplot for feature relationships
├── scatter3d_iris.py               # 3D scatter plot of selected features
├── README.md                       # Project documentation (this file)
```

---

## 📦 Dependencies

* Python 3.x
* pandas
* matplotlib
* seaborn
* scikit-learn

Install all required packages using:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## ▶️ How to Run Each Script

Each script is standalone. You can run them individually:

```bash
python histogram_plot.py
python boxplot_petal_length.py
python pairplot_iris.py
python scatter3d_iris.py
```

Make sure `iris.csv` is available in the same directory for the first three scripts.

---

## 🔍 Visualizations & Descriptions

---

### 📊 1. Histogram of Sepal Length

**Script:** `histogram_plot.py`

```python
plt.hist(df["sepal.length"], bins=10, color="skyblue", edgecolor="black")
```

* **Purpose:** Shows the frequency distribution of sepal lengths across all species.
* **Insight:** Helps identify common value ranges and data skewness.

---

### 📦 2. Boxplot of Petal Length by Variety

**Script:** `boxplot_petal_length.py`

```python
sns.boxplot(x="variety", y="petal.length", data=df)
```

* **Purpose:** Compares the distribution of petal lengths across species.
* **Insight:** Highlights class separability based on petal length (important for classification).

---

### 🔗 3. Pairplot of All Features

**Script:** `pairplot_iris.py`

```python
sns.pairplot(df, hue="variety")
```

* **Purpose:** Visualizes pairwise relationships between features.
* **Insight:** Makes it easy to identify which features best distinguish between species.

---

### 🌐 4. 3D Scatter Plot

**Script:** `scatter3d_iris.py`

```python
ax.scatter(x[:,2], x[:,3], x[:,0], c=y, cmap="viridis")
```

* **Purpose:** Projects 3 key features into a 3D space for better visual separation of classes.

* **Features Used:**

  * X-axis: Petal Length
  * Y-axis: Petal Width
  * Z-axis: Sepal Length

* **Insight:** Provides a spatial view of clustering between the three iris varieties.

---

## 📈 Sample Visual Outputs

While actual images aren't included here, running each script will open the respective plots. Here’s what you can expect:

* Histogram: A smooth distribution of sepal lengths.
* Boxplot: Three distinct box-and-whisker plots per variety.
* Pairplot: Grid of scatter plots with color-coded classes.
* 3D Scatter: Clear clusters in 3D for different varieties.

---

## 🧪 Learning Outcomes

Through this project, I demonstrate:

* Practical experience in **data visualization**
* Ability to perform **feature exploration**
* Use of tools like **Matplotlib**, **Seaborn**, and **Scikit-learn**
* Skills in analyzing **multidimensional relationships**

---

## 📧 Contact

* Email: kalokheyashshree@gmail.com
* GitHub: https://github.com/Yash1723vk/Graphs
