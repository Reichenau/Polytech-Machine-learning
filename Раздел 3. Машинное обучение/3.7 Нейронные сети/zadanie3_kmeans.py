import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = px.data.iris()

print("ЗАДАНИЕ 3: Метод k-средних (K-means)")
print("=" * 50)

print(f"Количество образцов в датасете: {len(df)}")
print("Распределение по классам:")
print(df['species'].value_counts())

# Подготавливаем данные (все признаки, все классы)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_true = df['species']

# Нормализуем данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nПризнаки для кластеризации: {X.columns.tolist()}")

# Определяем оптимальное количество кластеров методом локтя
inertias = []
silhouette_scores = []
k_range = range(2, 9)

print("\nПоиск оптимального количества кластеров:")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    print(f"k={k}: инерция={kmeans.inertia_:.2f}, силуэт={silhouette_avg:.3f}")

# Применяем k-means с k=3 (количество реальных классов)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X_scaled)

# Оцениваем качество кластеризации
ari_score = adjusted_rand_score(y_true, y_pred)
silhouette_avg = silhouette_score(X_scaled, y_pred)

print("\nРезультаты кластеризации с k=3:")
print(f"Adjusted Rand Index: {ari_score:.3f}")
print(f"Силуэтный коэффициент: {silhouette_avg:.3f}")

# Сравниваем предсказанные кластеры с реальными классами
print("\nСравнение кластеров с реальными классами:")
species_names = df['species'].unique()
for i in range(3):
    cluster_mask = y_pred == i
    cluster_species = y_true[cluster_mask].value_counts()
    print(f"Кластер {i}:")
    for species, count in cluster_species.items():
        print(f"  {species}: {count} образцов")

# Визуализация результатов
plt.figure(figsize=(15, 10))

# График 1: Метод локтя
plt.subplot(2, 3, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Инерция')
plt.title('Метод локтя')
plt.grid(True, alpha=0.3)

# График 2: Силуэтный анализ
plt.subplot(2, 3, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Силуэтный коэффициент')
plt.title('Силуэтный анализ')
plt.grid(True, alpha=0.3)

# График 3: Реальные классы
plt.subplot(2, 3, 3)
colors = ['red', 'blue', 'green']
for i, species in enumerate(species_names):
    mask = y_true == species
    data_subset = X[mask]
    plt.scatter(data_subset['sepal_length'], data_subset['sepal_width'], 
                c=colors[i], label=species, alpha=0.7)
plt.xlabel('Длина чашелистика')
plt.ylabel('Ширина чашелистика')
plt.title('Реальные классы')
plt.legend()
plt.grid(True, alpha=0.3)

# График 4: Предсказанные кластеры
plt.subplot(2, 3, 4)
for i in range(3):
    mask = y_pred == i
    data_subset = X[mask]
    plt.scatter(data_subset['sepal_length'], data_subset['sepal_width'], 
                c=colors[i], label=f'Кластер {i}', alpha=0.7)
plt.xlabel('Длина чашелистика')
plt.ylabel('Ширина чашелистика')
plt.title('Предсказанные кластеры')
plt.legend()
plt.grid(True, alpha=0.3)

# График 5: Центроиды кластеров
plt.subplot(2, 3, 5)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
for i in range(3):
    mask = y_pred == i
    data_subset = X[mask]
    plt.scatter(data_subset['petal_length'], data_subset['petal_width'], 
                c=colors[i], label=f'Кластер {i}', alpha=0.7)
    plt.scatter(centroids[i][2], centroids[i][3], 
                c='black', marker='x', s=200, linewidths=3)
plt.xlabel('Длина лепестка')
plt.ylabel('Ширина лепестка')
plt.title('Кластеры и центроиды')
plt.legend()
plt.grid(True, alpha=0.3)

# График 6: 3D визуализация
ax = plt.subplot(2, 3, 6, projection='3d')
for i in range(3):
    mask = y_pred == i
    data_subset = X[mask]
    ax.scatter(data_subset['sepal_length'], 
               data_subset['sepal_width'], 
               data_subset['petal_length'], 
               c=colors[i], label=f'Кластер {i}', alpha=0.7)
ax.set_xlabel('Длина чашелистика')
ax.set_ylabel('Ширина чашелистика')
ax.set_zlabel('Длина лепестка')
ax.set_title('3D визуализация кластеров')
ax.legend()

plt.tight_layout()
plt.show()

print("\nЦентроиды кластеров (в исходном масштабе):")
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for i, centroid in enumerate(centroids):
    print(f"Кластер {i}:")
    for j, value in enumerate(centroid):
        print(f"  {feature_names[j]}: {value:.3f}")
