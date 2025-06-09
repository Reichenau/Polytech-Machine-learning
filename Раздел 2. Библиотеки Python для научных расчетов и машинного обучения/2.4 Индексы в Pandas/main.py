import pandas as pd
import numpy as np

# # 1. Разобраться как использовать мультииндексные ключи в данном примере
print(f"{'Задание №1':-^40}")
index = [
    ('city_1', 2010),
    ('city_1', 2020),
    ('city_2', 2010),
    ('city_2', 2020),
    ('city_3', 2010),
    ('city_3', 2020),
]

index = pd.MultiIndex.from_tuples(index)


population = [
    101,
    201,
    102,
    202,
    103,
    203,
]
pop = pd.Series(population, index=index)
pop_df = pd.DataFrame(
    {
        'total': pop,
        'something': [
            10,
            11,
            12,
            13,
            14,
            15,
        ]
    }
)


pop_df_1 = pop_df.loc['city_1', 'something']
print(f"Пример 1 :\n{pop_df_1}\n")

pop_df_2 = pop_df.loc[['city_1', 'city_3'], ['total', 'something']]
print(f"Пример 2 :\n{pop_df_2}\n")


pop_df_3 = pop_df.loc[['city_1', 'city_3'], 'something']
print(f"Пример 3 :\n{pop_df_3}")

# 2. Из получившихся данных выбрать данные по 
print(f"{'Задание №2':-^40}")

index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020],
    ],
    names=['city', 'year'],
)

columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2'],
    ],
    names=['worker', 'job'],
)

rng = np.random.default_rng(1)
data = rng.random((4, 6))


data_df = pd.DataFrame(data, index=index, columns=columns)

print(f"Изначальный DataFrame:\n{data_df}")
# - 2020 году (для всех столбцов)
print("\n1. 2020 году (для всех столбцов)")
print(data_df.xs(2020, level='year'))

# - job_1 (для всех строк)
print("\n2. job_1 (для всех строк)")
print(data_df.xs("job_1", level='job', axis=1))

# - для city_1 и job_2 
print("\n3. для city_1 и job_2 ")
print(data_df.xs("city_1", level="city").xs("job_2", level="job", axis=1))

# 3. Взять за основу DataFrame со следующей структурой
print(f"{'Задание №3':-^40}")

index = pd.MultiIndex.from_product(
    [
        ['city_1', 'city_2'],
        [2010, 2020]
    ],
    names=['city', 'year']
)
columns = pd.MultiIndex.from_product(
    [
        ['person_1', 'person_2', 'person_3'],
        ['job_1', 'job_2']
    ],
    names=['worker', 'job']
)

rng = np.random.default_rng(1)
data = rng.random((4, 6))

data_df = pd.DataFrame(data, index=index, columns=columns)
print(f"Изначальный DataFrame:\n{data_df}")

# Выполнить запрос на получение следующих данных
# - все данные по person_1 и person_3
print("\n1. все данные по person_1 и person_3")
print(data_df.loc[:, (["person_1", "person_3"])])

# - все данные по первому городу и первым двум person-ам (с использование срезов)
print("\n2. все данные по первому городу и первым двум person-ам (с использованием срезов)")
print(data_df.loc['city_1', (["person_1", "person_2"])])

# Приведите пример (самостоятельно) с использованием pd.IndexSlice
print("\n3. Приведите пример (самостоятельно) с использованием pd.IndexSlice")
idx = pd.IndexSlice
print(data_df.loc[idx[:, 2010], idx[['person_1', 'person_3'], :]])

#4. Привести пример использования inner и outer джойнов для Series (данные примера скорее всего нужно изменить)
print(f"{'Задание №4':-^40}")

ser1 = pd.Series(['a', 'b', 'c'], index=[1, 2, 3])
ser2 = pd.Series(['b', 'c', 'f'], index=[2, 3, 4])

print("Series 1:")
print(ser1)

print("\nSeries 2:")
print(ser2)

print("\nOuter Join:")
print(pd.concat([ser1, ser2], axis=1, join='outer'))

print("\nInner Join:")
print(pd.concat([ser1, ser2], axis=1, join='inner'))
