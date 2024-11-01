import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations


def load_iris_data(file_path):
    """Đọc dữ liệu từ file iris.data"""
    data = []
    labels = []
    label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                values = line.strip().split(',')
                if len(values) == 5:
                    features = [float(x) for x in values[:4]]
                    label = label_map[values[4]]
                    data.append(features)
                    labels.append(label)

    return np.array(data), np.array(labels)


def euclidean_distance(x1, x2):
    """Tính khoảng cách Euclidean không dùng hàm có sẵn"""
    sum_sq = 0
    for i in range(len(x1)):
        sum_sq += (x1[i] - x2[i]) ** 2
    return sum_sq ** 0.5


def initialize_centroids(X, k):
    """Khởi tạo centroids ngẫu nhiên"""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)[:k]
    return X[indices].copy()


def assign_clusters(X, centroids):
    """Gán cluster cho từng điểm dữ liệu"""
    n_samples = len(X)
    labels = np.zeros(n_samples, dtype=int)
    distances = np.zeros(n_samples)

    for i in range(n_samples):
        min_dist = float('inf')
        for j, centroid in enumerate(centroids):
            dist = euclidean_distance(X[i], centroid)
            if dist < min_dist:
                min_dist = dist
                labels[i] = j
        distances[i] = min_dist

    return labels, np.sum(distances)


def update_centroids(X, labels, k):
    """Cập nhật vị trí centroids"""
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    counts = np.zeros(k)

    # Tính tổng các điểm trong mỗi cluster
    for i in range(len(X)):
        cluster = int(labels[i])
        for j in range(n_features):
            centroids[cluster][j] += X[i][j]
        counts[cluster] += 1

    # Tính trung bình
    for i in range(k):
        if counts[i] > 0:
            centroids[i] = centroids[i] / counts[i]

    return centroids


def kmeans(X, k, max_iters=100):
    """Thuật toán K-means"""
    centroids = initialize_centroids(X, k)
    best_labels = None
    best_inertia = float('inf')

    for _ in range(max_iters):
        old_centroids = centroids.copy()
        labels, inertia = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, k)

        # Lưu kết quả tốt nhất
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

        # Kiểm tra điều kiện dừng
        if np.all(np.abs(old_centroids - centroids) < 1e-6):
            break

    return best_labels, centroids, best_inertia


def calculate_f1_score(true_labels, pred_labels, k):
    """Tính F1-score thủ công"""
    best_f1 = 0

    for perm in permutations(range(k)):
        mapped_labels = np.zeros_like(pred_labels)
        for i, p in enumerate(perm):
            mapped_labels[pred_labels == i] = p

        f1_scores = []
        for i in range(k):
            # Tính các metrics thủ công
            true_pos = sum((true_labels == i) & (mapped_labels == i))
            false_pos = sum((true_labels != i) & (mapped_labels == i))
            false_neg = sum((true_labels == i) & (mapped_labels != i))

            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores)
        if avg_f1 > best_f1:
            best_f1 = avg_f1

    return best_f1


def plot_results(X, labels, k, title):
    """Vẽ kết quả clustering"""
    plt.figure(figsize=(15, 5))

    # Plot features 1 vs 2
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.title(f'{title}\nSepal Length vs Sepal Width')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')

    # Plot features 3 vs 4
    plt.subplot(132)
    plt.scatter(X[:, 2], X[:, 3], c=labels, cmap='viridis')
    plt.title('Petal Length vs Petal Width')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')

    # Plot features 1 vs 3
    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 2], c=labels, cmap='viridis')
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Đọc dữ liệu
    X, y = load_iris_data('iris.data')

    # Chuẩn hóa dữ liệu
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    X_normalized = (X - means) / stds

    k_values = [2, 3, 4, 5]
    results = []

    for k in k_values:

        best_f1 = 0
        best_labels = None
        best_inertia = float('inf')

        for _ in range(5):
            labels, centroids, inertia = kmeans(X_normalized, k)
            f1 = calculate_f1_score(y, labels, k)

            if f1 > best_f1:
                best_f1 = f1
                best_labels = labels
                best_inertia = inertia

        results.append({
            'k': k,
            'f1_score': best_f1,
            'inertia': best_inertia,
            'labels': best_labels
        })

        # Vẽ kết quả cho mỗi k
        plot_results(X, best_labels, k, f'K-means Clustering (k={k})')

    # In kết quả
    print("\nKết quả phân cụm với các giá trị k khác nhau:")
    print("=" * 50)
    for result in results:
        print(f"k = {result['k']}:")
        print(f"  F1-score: {result['f1_score']:.4f}")
        print(f"  Inertia: {result['inertia']:.4f}")
        print("-" * 30)

    # Vẽ biểu đồ so sánh F1-score và Inertia
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(k_values, [r['f1_score'] for r in results], 'bo-')
    plt.title('F1-score vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('F1-score')

    plt.subplot(122)
    plt.plot(k_values, [r['inertia'] for r in results], 'ro-')
    plt.title('Inertia vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')

    plt.tight_layout()
    plt.show()
