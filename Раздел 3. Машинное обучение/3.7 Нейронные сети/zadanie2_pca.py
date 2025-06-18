import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = px.data.iris()

print("ЗАДАНИЕ 2: Метод главных компонент (PCA)")
print("=" * 50)

# Выбираем только два сорта: setosa и virginica
df_filtered = df[df['species'].isin(['setosa', 'virginica'])].copy()

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

# Применяем PCA для снижения размерности
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nОбъясненная дисперсия первой компоненты: "
      f"{pca.explained_variance_ratio_[0]:.3f}")
print(f"Объясненная дисперсия второй компоненты: "
      f"{pca.explained_variance_ratio_[1]:.3f}")
print(f"Общая объясненная дисперсия: "
      f"{sum(pca.explained_variance_ratio_):.3f}")

# Обучаем модель классификации на данных после PCA
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train_pca, y_train)

# Делаем предсказания
y_pred = classifier.predict(X_test_pca)

# Оцениваем точность
accuracy = accuracy_score(y_test, y_pred)
print(f"\nТочность модели после PCA: {accuracy:.3f}")

print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

# Визуализация результатов
plt.figure(figsize=(15, 5))

# График 1: Исходные данные в пространстве главных компонент
plt.subplot(1, 3, 1)
for species in df_filtered['species'].unique():
    mask = y_train == species
    plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], 
                label=species, alpha=0.7)
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('Обучающие данные в пространстве PCA')
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: Результаты классификации
plt.subplot(1, 3, 2)
for species in ['setosa', 'virginica']:
    mask = y_pred == species
    if mask.any():
        plt.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], 
                    label=f'Предсказано: {species}', alpha=0.7)
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('Результаты классификации')
plt.legend()
plt.grid(True, alpha=0.3)

# График 3: Объясненная дисперсия
plt.subplot(1, 3, 3)
components = range(1, len(pca.explained_variance_ratio_) + 1)
plt.bar(components, pca.explained_variance_ratio_)
plt.xlabel('Номер компоненты')
plt.ylabel('Объясненная дисперсия')
plt.title('Объясненная дисперсия по компонентам')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nКомпоненты PCA (веса признаков):")
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for i, component in enumerate(pca.components_):
    print(f"Компонента {i+1}:")
    for j, weight in enumerate(component):
        print(f"  {feature_names[j]}: {weight:.3f}")
