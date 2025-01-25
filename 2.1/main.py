import numpy as np
import array
import sys

## 1. Какие еще существуют коды типов?
# signed char 'b'
a1 = array.array('b', [-128, 0, 127])
# unsigned char 'B'
a2 = array.array('B', [0, 128, 255])
# signed short 'h'
a3 = array.array('h', [-32768, 0, 32767])
# unsigned short 'H'
a4 = array.array('H', [0, 32768, 65535])
# signed int 'i'
a5 = array.array('i', [-2147483648, 0, 2147483647])
# unsigned int 'I'
a6 = array.array('I', [0, 2147483648, 4294967295])
# signed long 'l'
a7 = array.array('l', [-2147483648, 0, 2147483647])
# unsigned long 'L'
a8 = array.array('L', [0, 2147483648, 4294967295])
# signed long long 'q'
a9 = array.array('q', [-9223372036854775808, 0, 9223372036854775807])
# unsigned long long 'Q'
a10 = array.array('Q', [0, 9223372036854775808, 18446744073709551615])
# float 'f'
a11 = array.array('f', [1.23, -45.67, 0.0])
# double 'd'
a12 = array.array('d', [1.23, -45.67, 0.0])

## 2. Напишите код, подобный приведенному выше, но с другим типом
a1 = array.array('B', [1, 2, 3])
print(sys.getsizeof(a1))
print(type(a1))

a2 = array.array('h', [-100, 0, 100])
print(sys.getsizeof(a2))
print(type(a2))

a3 = array.array('d', [1.23, 4.56, 7.89])
print(sys.getsizeof(a3))
print(type(a3))

## 3. Напишите код для создания массива с 5 значениями, располагающимися через равные интервалы в диапазоне от 0 до 1
arr = np.arange(0, 1.25, 0.25) 
print(arr)

## 4. Напишите код для создания массива с 5 равномерно распределенными случайными значениями в диапазоне от 0 до 1
arr = np.random.uniform(0, 1.25, size=5)
print(arr)

## 5. Напишите код для создания массива с 5 нормально распределенными случайными значениями с мат. ожиданием = 0 и дисперсией 1
arr = np.random.normal(loc=0, scale=1, size=5)
print(arr)

## 6. Напишите код для создания массива с 5 случайными целыми числами в от [0, 10)
arr = np.random.randint(0, 10, 5)
print(arr)

## 7. Написать код для создания срезов массива 3 на 4
arr = arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

## - первые две строки и три столбца
print(arr[:2, :3])

## - первые три строки и второй столбец
print(arr[:3, 1:2])

## - все строки и столбцы в обратном порядке
print(arr[:, ::-1])

## - второй столбец
print(arr[:, 1])

## - третья строка
print(arr[2, :])

## 8. Продемонстрируйте, как сделать срез-копию
arr_old = arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
arr_new = arr_old[:2, :3]
print(arr_new)

## 9. Продемонстрируйте использование newaxis для получения вектора-столбца и вектора-строки
arr = np.array([1, 2, 3, 4])

column_vector = arr[:, np.newaxis]
print(column_vector)

row_vector = arr[np.newaxis, :]
print(row_vector)

## 10. Разберитесь, как работает метод dstack
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
result = np.dstack((a, b))
print(result)

## 11. Разберитесь, как работают методы split, vsplit, hsplit, dsplit
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.split(arr, 3,))
print(np.vsplit(arr, 3))
print(np.hsplit(arr, 3))
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(np.dsplit(arr, 3))
## 12. Привести пример использования всех универсальных функций, которые я привел