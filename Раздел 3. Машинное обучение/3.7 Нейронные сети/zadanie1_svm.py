import plotly.express as px
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = px.data.iris()

print("ЗАДАНИЕ 1: Метод опорных векторов (SVM)")
print("=" * 50)

# Выбираем только два сорта: setosa и versicolor
df_filtered = df[df['species'].isin(['setosa', 'versicolor'])].copy()

print(f"Количество образцов после фильтрации: {len(df_filtered)}")
print("Распределение по классам:")
print(df_filtered['species'].value_counts())

# Подготавливаем данные
X = df_filtered[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df_filtered['species']

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Нормализуем данные
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nРазмер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

# Создаем и обучаем модель SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Делаем предсказания
y_pred = svm_model.predict(X_test_scaled)

# Оцениваем точность
accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели SVM: {accuracy:.3f}")

print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

# Визуализация результатов (2D проекция)
plt.figure(figsize=(12, 5))

# График 1: Исходные данные
plt.subplot(1, 2, 1)
for species in df_filtered['species'].unique():
    data = df_filtered[df_filtered['species'] == species]
    plt.scatter(data['sepal_length'], data['sepal_width'], 
                label=species, alpha=0.7)
plt.xlabel('Длина чашелистика')
plt.ylabel('Ширина чашелистика')
plt.title('Исходные данные (setosa vs versicolor)')
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: Результаты предсказания
plt.subplot(1, 2, 2)
X_test_df = pd.DataFrame(X_test, columns=X.columns)
for i, species in enumerate(['setosa', 'versicolor']):
    mask = y_pred == species
    if mask.any():
        scatter_data = X_test_df[mask]
        plt.scatter(scatter_data['sepal_length'], 
                   scatter_data['sepal_width'], 
                   label=f'Предсказано: {species}', alpha=0.7)

plt.xlabel('Длина чашелистика')
plt.ylabel('Ширина чашелистика')
plt.title('Результаты классификации SVM')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
