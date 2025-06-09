import numpy as np
import pandas as pd

# 1. Привести различные способы создания объектов типа Series
print(f"{'Задание №1':-^40}")

# Для создания Series можно использовать
# - списки Python или массивы NumPy
print("Series из списков Python:")
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
series_from_list = pd.Series(data)
print(series_from_list)

print("\nSeries из массивов NumPy:")
array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
series_from_array = pd.Series(array)
print(series_from_array)

# - скалярные значение
print("\nSeries из скалярных значений:")
scalar = 100
series_from_scalar = pd.Series(scalar)
print(series_from_scalar)

# - словари
print("\nSeries из словарей:")
dict = {
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
}
series_from_dict = pd.Series(dict)
print(series_from_dict)

# 2. Привести различные способы создания объектов типа DataFrame
print(f"{'Задание №2':-^40}")
# DataFrame. Способы создания
# - через объекты Series
print("\nDataFrame из объектов Series:")
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['a', 'b', 'c'])

df_from_series = pd.DataFrame({
    'column_1': s1,
    'column_2': s2,
})
print(df_from_series)

# - списки словарей
print("\nDataFrame из словарей:")
data = [
    {
        'a': 1,
        'b': 2,
    },
    {
        'a': 5,
        'b': 10,
        'c': 20,
    }
]
df_from_list_of_dict = pd.DataFrame(data)
print(df_from_list_of_dict)

# - словари объектов Series
print("\nDataFrame из словарей объектов Series:")
data = {
    'I': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
    'II': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
    'III': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
}
df_from_dict_of_series = pd.DataFrame(data)
print(df_from_dict_of_series)

# - двумерный массив NumPy
print("\nDataFrame из двумерных массивов NumPy:")
array = np.arange(1, 10).reshape(3, 3)
df_from_numpy_array = pd.DataFrame(array)
print(df_from_numpy_array)

# - структурированный массив Numpy
print("\nDataFrame из структурированного массива NumPy:")

# 3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так, чтобы вместо NaN было установлено значение 1
print(f"{'Задание №3':-^40}")

population_dict = {
    'city_1': 1001,
    'city_2': 1002,
    'city_3': 1003,
    'city_41': 1004,
    'city_51': 1005,
}
area_dict = {
    'city_1': 9991,
    'city_2': 9992,
    'city_3': 9993,
    'city_42': 9994,
    'city_52': 9995,
}

population = pd.Series(population_dict)
area = pd.Series(area_dict)

data = pd.DataFrame({
    'area1': area,
    'population1': population,
})
data = data.fillna(1)
print(data)

# 4. Переписать пример с транслирование для DataFrame так, чтобы вычитание происходило по СТОЛБЦАМ
print(f"{'Задание №4':-^40}")
rng = np.random.default_rng(1)

A = rng.integers(0, 10, (3, 4))

df = pd.DataFrame(A, columns=['a', 'b', 'c', 'd'])
print(df.sub(df['a'], axis=0))  # вычитание первого столбца из всех эементов 

# 5. На примере объектов DataFrame продемонстрируйте использование методов ffill() и bfill()
print(f"{'Задание №5':-^40}")
data = {
    'A': [1, 2, np.nan, 4, np.nan],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, np.nan, np.nan, 4, 5]
}

df = pd.DataFrame(data)
print(df)

print("\nDataFrame метод ffill:")
df_ffill = df.ffill()
print(df_ffill)

print("\nDataFrame метод bfill:")
df_bfill = df.bfill()
print(df_bfill)
