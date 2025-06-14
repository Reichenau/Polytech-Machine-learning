# Практическое задание
# убрать из данных iris часть точек (на которых мы обучаемся ) и убедиться что на предсказание влияют только опорные вектора.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

iris = sns.load_dataset("iris")
data = iris[["sepal_length", "petal_length", "species"]]
data_df = data[
    (data["species"] == "setosa") | (data["species"] == "versicolor")
]
X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

model_full = SVC(kernel='linear', C=10_000)
model_full.fit(X, y)
support_indices = model_full.support_

X_support = X.iloc[support_indices]
y_support = y.iloc[support_indices]


data_df = pd.DataFrame({
    'sepal_length': X_support['sepal_length'],
    'petal_length': X_support['petal_length'],
    'species': y_support
})

X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

data_df_setosa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

plt.scatter(
    data_df_setosa["sepal_length"],
    data_df_setosa["petal_length"],
)
plt.scatter(
    data_df_versicolor["sepal_length"],
    data_df_versicolor["petal_length"],
)

model = SVC(kernel='linear', C=10_000)
model.fit(X, y)

plt.scatter(model.support_vectors_[:,0], 
            model.support_vectors_[:,1], 
            s=400, 
            facecolor='none', 
            edgecolors="black")

x1_p = np.linspace(
    data_df["sepal_length"].min() - 1, data_df["sepal_length"].max() + 1, 100
)
x2_p = np.linspace(
    data_df["petal_length"].min() - 1, data_df["petal_length"].max() + 1, 100
)
x1_p, x2_p = np.meshgrid(x1_p, x2_p)
X_p = pd.DataFrame(
    np.c_[x1_p.ravel(), x2_p.ravel()],
    columns=["sepal_length", "petal_length"]
)
y_p = model.predict(X_p)
X_p["species"] = y_p
plt.scatter(
    X_p[X_p["species"] == "setosa"]["sepal_length"],
    X_p[X_p["species"] == "setosa"]["petal_length"],
    alpha=0.4
)
plt.scatter(
    X_p[X_p["species"] == "versicolor"]["sepal_length"],
    X_p[X_p["species"] == "versicolor"]["petal_length"],
    alpha=0.4
)
plt.show()
