import os
import glob
import random
import shutil
import time
from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from sklearn.cluster import AffinityPropagation, Birch, MeanShift, OPTICS, SpectralClustering
from sklearn.cluster import SpectralBiclustering, HDBSCAN
#import hdbscan

# Define paths
#data_paths = ['slider_based_shapes_data_exif_neil_1', 'slider_based_shapes_data_melanie_1']
data_paths = ['slider_based_shapes_data_melanie_1']
timestamp_global = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f'results_{timestamp_global}'
os.makedirs(results_dir, exist_ok=True)

def load_images_from_folders(paths):
    images, labels = [], []
    for label, path in enumerate(paths):
        for filename in glob.glob(os.path.join(path, '*.png')):
            img = load_img(filename, target_size=(28, 28), color_mode='grayscale')
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def build_and_train_mnist_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=128)
    return model

def extract_embeddings(model, images):
    intermediate_layer_model = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    embeddings = intermediate_layer_model.predict(images)
    return embeddings

def plot_elbow_method(data):
    distortions = []
    K = range(2, 12)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        distortions.append(kmeans.inertia_)
    plt.figure()
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    elbow_path = os.path.join(results_dir, f'elbow_kmeans_{timestamp}.png')
    plt.savefig(elbow_path)
    return distortions.index(min(distortions)) + 2  # +2 because we started at k=2

def tune_dbscan(data):
    best_score = -1
    best_params = {'eps': 0.5, 'min_samples': 5}
    for eps in np.arange(0.5, 5.0, 0.5):
        for min_samples in range(3, 10):
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(data)
            if len(set(labels)) > 1 and -1 not in labels:
                try:
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                except:
                    continue
    return best_params

def run_clustering(algo_name, data):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if algo_name == 'kmeans':
        optimal_k = plot_elbow_method(data)
        model = KMeans(n_clusters=optimal_k, random_state=42).fit(data)
    elif algo_name == 'dbscan':
        best_params = tune_dbscan(data)
        model = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples']).fit(data)
    elif algo_name == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=2).fit(data)
    else:
        raise ValueError("Unsupported clustering method")

    labels = model.labels_
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
    plt.title(f'Clustering with {algo_name}')
    plot_path = os.path.join(results_dir, f'{algo_name}_clusters_{timestamp}.png')
    plt.savefig(plot_path)

    # Save interactive plot
    fig = px.scatter(x=data[:, 0], y=data[:, 1], color=labels.astype(str),
                     title=f'Interactive Clustering - {algo_name}', labels={'x': 'Component 1', 'y': 'Component 2'})
    interactive_path = os.path.join(results_dir, f'{algo_name}_interactive_{timestamp}.html')
    pio.write_html(fig, file=interactive_path, auto_open=False)

    return labels, plot_path, timestamp

def generate_cluster_summary(cluster_labels, true_labels, algo_name, timestamp, data):
    summary = {}
    unique_labels = set(cluster_labels)
    for lbl in unique_labels:
        summary[lbl] = np.sum(cluster_labels == lbl)

    text_path = os.path.join(results_dir, f'{algo_name}_summary_{timestamp}.txt')
    with open(text_path, 'w') as f:
        for key in summary:
            f.write(f'Cluster {key}: {summary[key]} images\n')

        matrix = confusion_matrix(true_labels, cluster_labels)
        acc = np.sum(np.max(matrix, axis=0)) / np.sum(matrix)
        f.write(f'Accuracy: {acc:.4f}\n')

        if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
            sil_score = silhouette_score(data, cluster_labels)
            f.write(f'Silhouette Score: {sil_score:.4f}\n')
        else:
            sil_score = None
            f.write('Silhouette Score: N/A\n')

    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {algo_name}')
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Label')
    cm_path = os.path.join(results_dir, f'{algo_name}_confusion_matrix_{timestamp}.png')
    plt.savefig(cm_path)

    return summary, acc, sil_score

def save_cluster_images(images, cluster_labels, algo_name, timestamp):
    unique_clusters = np.unique(cluster_labels)
    selected_images = []
    for cluster_id in unique_clusters:
        idxs = np.where(cluster_labels == cluster_id)[0]
        if len(idxs) == 0:
            continue
        idx = random.choice(idxs)
        img = images[idx].squeeze()
        selected_images.append((img, cluster_id))

    fig, axes = plt.subplots(1, len(selected_images), figsize=(15, 5))
    for ax, (img, cluster_id) in zip(axes, selected_images):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Cluster {cluster_id}')
        ax.axis('off')
    super_img_path = os.path.join(results_dir, f'{algo_name}_super_image_{timestamp}.png')
    plt.savefig(super_img_path)

# Load data and train model
images, true_labels = load_images_from_folders(data_paths)
images = images.reshape((-1, 28, 28, 1))
trained_model = build_and_train_mnist_model()

# Extract embeddings
embeddings = extract_embeddings(trained_model, images)

# Reduce dimensions for clustering visualization
try:
    reducer = PCA(n_components=2)
    embeddings_2d = reducer.fit_transform(embeddings)
    method_used = "pca"
except Exception as e:
    print("PCA failed, using t-SNE")
    reducer = TSNE(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    method_used = "tsne"

plt.figure()
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=true_labels, cmap='viridis', s=5)
plt.title(f'{method_used.upper()} Visualization of Embeddings')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
projection_path = os.path.join(results_dir, f'embedding_{method_used}_{timestamp}.png')
plt.savefig(projection_path)

clustering_algorithms = {
    'kmeans': KMeans(n_clusters=2, random_state=42),
    'dbscan': DBSCAN(eps=0.5, min_samples=5),
    'hierarchical': AgglomerativeClustering(n_clusters=2),
    'affinity': AffinityPropagation(random_state=42),
    'birch': Birch(n_clusters=2),
    'hdbscan': HDBSCAN(min_cluster_size=5),
    'meanshift': MeanShift(),
    'optics': OPTICS(min_samples=5),
    'spectral_biclustering': SpectralBiclustering(n_clusters=2, random_state=42)
}

results = {}
for algo_name, algo_model in clustering_algorithms.items():
    try:
        labels = algo_model.fit_predict(embeddings_2d)
        plt.figure()
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='rainbow', s=5)
        plt.title(f'{algo_name.upper()} Clustering')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(results_dir, f'{algo_name}_clustering_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()

        # Save interactive plot
        fig = px.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], color=labels.astype(str),
                         title=f'Interactive Clustering - {algo_name}', labels={'x': 'Component 1', 'y': 'Component 2'})
        interactive_path = os.path.join(results_dir, f'{algo_name}_interactive_{timestamp}.html')
        pio.write_html(fig, file=interactive_path, auto_open=False)

        summary, accuracy, sil_score = generate_cluster_summary(labels, true_labels, algo_name, timestamp,
                                                                embeddings_2d)
        save_cluster_images(images, labels, algo_name, timestamp)
        results[algo_name] = {
            'summary': summary,
            'accuracy': accuracy,
            'silhouette_score': sil_score,
            'plot_path': save_path
        }


    except Exception as e:
        print(f"{algo_name} failed with error: {e}")


    # labels = model.labels_
    # plt.figure()
    # plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow')
    # plt.title(f'Clustering with {algo_name}')
    # plot_path = os.path.join(results_dir, f'{algo_name}_clusters_{timestamp}.png')
    # plt.savefig(plot_path)
    #
    # # Save interactive plot
    # fig = px.scatter(x=data[:, 0], y=data[:, 1], color=labels.astype(str),
    #                  title=f'Interactive Clustering - {algo_name}', labels={'x': 'Component 1', 'y': 'Component 2'})
    # interactive_path = os.path.join(results_dir, f'{algo_name}_interactive_{timestamp}.html')
    # pio.write_html(fig, file=interactive_path, auto_open=False)
    #
    # return labels, plot_path, timestamp



# results = {}
# for algo in ['kmeans', 'dbscan', 'hierarchical']:
#     cluster_labels, plot_path, timestamp = run_clustering(algo, embeddings_2d)
#     summary, accuracy, sil_score = generate_cluster_summary(cluster_labels, true_labels, algo, timestamp, embeddings_2d)
#     save_cluster_images(images, cluster_labels, algo, timestamp)
#     results[algo] = {
#         'summary': summary,
#         'accuracy': accuracy,
#         'silhouette_score': sil_score,
#         'plot_path': plot_path
#     }

with open(os.path.join(results_dir, 'clustering_results_summary.txt'), 'w') as f:
    for algo, result in results.items():
        f.write(f"Algorithm: {algo}\nAccuracy: {result['accuracy']:.4f}\n")
        if result['silhouette_score'] is not None:
            f.write(f"Silhouette Score: {result['silhouette_score']:.4f}\n")
        else:
            f.write("Silhouette Score: N/A\n")
        for cluster_id, count in result['summary'].items():
            f.write(f"  Cluster {cluster_id}: {count} images\n")
        f.write("\n")

csv_path = os.path.join(results_dir, 'clustering_results_summary.csv')
with open(csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Algorithm', 'Accuracy', 'Silhouette Score', 'Cluster ID', 'Image Count'])
    for algo, result in results.items():
        for cluster_id, count in result['summary'].items():
            writer.writerow([
                algo,
                f"{result['accuracy']:.4f}",
                f"{result['silhouette_score']:.4f}" if result['silhouette_score'] is not None else 'N/A',
                cluster_id,
                count
            ])
