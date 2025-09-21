import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

def cancerdetection():
    cancer = load_breast_cancer()

    x = cancer.data
    y = cancer.target
    data = pd.DataFrame(x, columns = cancer.feature_names)
    data['target'] = y

    print(data.shape)
    print("classes: ", dict(zip(cancer.target_names, [0, 1])))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    model = SVC(kernel = 'linear', C=1)   # c helps to find the margin properly
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy is: ",accuracy)

    print("\nconfusion matrix: ", confusion_matrix(y_test, y_pred))

def main():
    cancerdetection()

if __name__ == "__main__":
    main()
