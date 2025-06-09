import numpy as np
## 1. Что надо изменить в последнем примере, чтобы он заработал без ошибок (транслирование)?

a = np.ones((3, 2))
b = np.arange(3)

b = b[:, np.newaxis]  

c = a + b
print(c)
## 2. Пример для y. Вычислить количество элементов (по обоим размерностям), значения которых больше 3 и меньше 9

y = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
result_axis1 = np.sum((y > 3) & (y < 9), axis=1)
result_axis0 = np.sum((y > 3) & (y < 9), axis=0)
total_result = np.sum((y > 3) & (y < 9))
print("Количество элементов по строкам (axis=1):", result_axis1)
print("Количество элементов по столбцам (axis=0):", result_axis0)
print("Общее количество элементов:", total_result)