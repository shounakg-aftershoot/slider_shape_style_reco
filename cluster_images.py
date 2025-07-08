import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont  # Added ImageDraw, ImageFont for labeling
from datetime import datetime  # For generating timestamps
import random  # For picking random images from clusters
from collections import Counter  # For counting elements in clusters

# --- Configuration ---
DATA_FOLDER_1 = 'slider_based_shapes_data_kishore_profile1'  # Updated folder name
DATA_FOLDER_2 = 'slider_based_shapes_data_kishore_profile2'  # Updated folder name
RESULTS_FOLDER = 'results'  # Folder to store all results
IMAGE_SIZE = (28, 28)  # MNIST image size
NUM_CLASSES = 10  # Number of classes in MNIST
K_RANGE = range(2, 12)  # Range for K in Elbow Method (2 to 11 inclusive)
EPOCHS = 5  # Number of epochs for training the MNIST model (can be increased for better performance)
BATCH_SIZE = 32
REP_IMAGE_DISPLAY_SIZE = (96, 96)  # Size for representative images in the super image (larger to fit text)

# DBSCAN specific parameters (may need tuning for your dataset)
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5


# --- Helper function for generating timestamped filenames ---
def get_timestamp_filename(prefix, algo_name, extension):
    """Generates a filename with current timestamp, algorithm name, and includes results folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(RESULTS_FOLDER, f"{prefix}_{algo_name}_{timestamp}.{extension}")


# --- Helper function to add text to an image (for representative images) ---
def add_text_to_image(img_pil, text, font_size=18, text_color=(255, 255, 255), bg_color=(0, 0, 0, 128)):
    """Adds text to a PIL Image."""
    draw = ImageDraw.Draw(img_pil)
    try:
        # Try to load a default font (e.g., Arial on Windows/macOS, LiberationSans on Linux)
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # Fallback to a simpler default font if system font is not found
        font = ImageFont.load_default()
        print("Warning: Arial font not found, using default font for labels.")

    # Get text bounding box for background calculation
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:  # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(text, font=font)

    # Position the text at the bottom left with some padding
    text_x = 5
    text_y = img_pil.height - text_height - 5

    # Draw a semi-transparent black background rectangle for better readability
    draw.rectangle([text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2], fill=bg_color)
    draw.text((text_x, text_y), text, font=font, fill=text_color)
    return img_pil


# --- Create results folder if it doesn't exist ---
os.makedirs(RESULTS_FOLDER, exist_ok=True)
print(f"Results will be saved in: '{RESULTS_FOLDER}'")

# --- 1. Train a simple CNN on MNIST (as a 'pre-trained' model) ---
print("\n--- Step 1: Training a simple CNN on MNIST ---")

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape and normalize images
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# One-hot encode labels
train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)

# Define the CNN model
mnist_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(name='flatten_layer'),
    layers.Dense(64, activation='relu', name='embedding_layer'),
    layers.Dense(NUM_CLASSES, activation='softmax', name='output_layer')
])

mnist_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

print(mnist_model.summary())

# Train the model
history = mnist_model.fit(train_images, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE,
                          validation_data=(test_images, test_labels),
                          verbose=0)  # verbose=0 to suppress training output

test_loss, test_acc = mnist_model.evaluate(test_images, test_labels, verbose=2)
print(f"MNIST Test Accuracy: {test_acc}")

# --- 2. Create a feature extraction model from the pre-trained model ---
print("\n--- Step 2: Creating a feature extraction model ---")

feature_extractor = models.Model(inputs=mnist_model.inputs,
                                 outputs=mnist_model.get_layer('embedding_layer').output)
print("Feature extractor model created from 'embedding_layer'.")

# --- 3. Load and preprocess your custom dataset ---
print("\n--- Step 3: Loading and preprocessing custom dataset ---")


def load_and_preprocess_image_for_model(image_path):
    """Loads an image, converts to grayscale, resizes, and normalizes for model input."""
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize(IMAGE_SIZE)  # Resize to MNIST input size
        img_array = np.array(img).astype('float32') / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (28, 28, 1)
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


all_images = []
all_image_paths = []


# Function to load images from a given directory
def load_images_from_folder(folder_path):
    print(f"Searching for images in: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist.")
        return

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                processed_img = load_and_preprocess_image_for_model(image_path)
                if processed_img is not None:
                    all_images.append(processed_img)
                    all_image_paths.append(image_path)


load_images_from_folder(DATA_FOLDER_1)
load_images_from_folder(DATA_FOLDER_2)

if not all_images:
    print("No images found in the specified folders. Please check your paths and image files.")
    exit()

dataset_images = np.array(all_images)
print(f"Loaded {len(dataset_images)} images from custom dataset.")

# --- 4. Extract embeddings for your dataset ---
print("\n--- Step 4: Extracting embeddings ---")
embeddings = feature_extractor.predict(dataset_images)
print(f"Extracted embeddings with shape: {embeddings.shape}")


# --- Function to perform clustering, plot, generate stats, and super image ---
def process_clustering_results(embeddings, all_image_paths, clusters, algo_name, optimal_k_for_viz=None):
    """
    Performs plotting, statistics generation, and super image creation for a given clustering result.
    optimal_k_for_viz is used for plot titles/stats if algo_name is not K-Means (e.g., for Agglomerative).
    """
    num_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)  # Exclude noise cluster from count
    print(f"\n--- Processing results for {algo_name} ---")
    print(f"Number of clusters found: {num_clusters_found}")

    # --- Visualize the clusters using PCA ---
    print(f"Visualizing clusters for {algo_name} with PCA...")
    embeddings_2d = embeddings
    if embeddings.shape[1] > 2:  # Only apply PCA if embedding dimension is greater than 2
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        print(f"  Reduced embeddings to 2D using PCA. Explained variance ratio: {pca.explained_variance_ratio_}")

    plt.figure(figsize=(10, 8))
    # Handle noise points in DBSCAN visualization
    if algo_name == 'DBSCAN' and -1 in clusters:
        # Plot noise points in black
        noise_mask = (clusters == -1)
        plt.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1],
                    c='black', s=30, alpha=0.5, label='Noise')
        # Plot actual clusters
        non_noise_mask = (clusters != -1)
        scatter = plt.scatter(embeddings_2d[non_noise_mask, 0], embeddings_2d[non_noise_mask, 1],
                              c=clusters[non_noise_mask], cmap='viridis', s=50, alpha=0.7)
        plt.legend()
    else:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=clusters, cmap='viridis', s=50, alpha=0.7)

    plot_title_k = optimal_k_for_viz if optimal_k_for_viz else num_clusters_found
    plt.title(
        f'Clustering of Custom Dataset Images ({algo_name} - K={plot_title_k if optimal_k_for_viz else num_clusters_found})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)

    cluster_plot_filename = get_timestamp_filename("cluster_plot", algo_name.lower().replace(" ", "_"), "png")
    plt.savefig(cluster_plot_filename)
    print(f"  Cluster plot saved as '{cluster_plot_filename}'")
    plt.close()  # Close plot to free memory

    # --- Generate and Save Cluster Statistics ---
    print(f"Generating and Saving Cluster Statistics for {algo_name}...")

    cluster_counts_dict = Counter(clusters)
    # Prepare content for console and file
    stats_output = [f"\n--- Cluster Statistics for {algo_name} ---"]
    stats_output.append(f"Number of clusters found: {num_clusters_found}")
    stats_output.append(f"{'Cluster ID':<12} {'Number of Images':<18}")
    stats_output.append("-" * 30)

    # Sort cluster IDs for consistent output, put -1 (noise) at the end
    sorted_cluster_ids = sorted(cluster_counts_dict.keys())
    if -1 in sorted_cluster_ids:
        sorted_cluster_ids.remove(-1)
        sorted_cluster_ids.append(-1)  # Move noise to end

    for cluster_id in sorted_cluster_ids:
        count = cluster_counts_dict[cluster_id]
        display_id = "Noise (-1)" if cluster_id == -1 else str(cluster_id)
        stats_output.append(f"{display_id:<12} {count:<18}")
    stats_output.append("-" * 30)

    # Print to console
    for line in stats_output:
        print(line)

    # Save to a text file
    stats_filename = get_timestamp_filename("cluster_stats", algo_name.lower().replace(" ", "_"), "txt")
    with open(stats_filename, 'w') as f:
        for line in stats_output:
            f.write(line + '\n')
    print(f"  Cluster statistics saved to '{stats_filename}'")

    # --- Generate Super Image with Representative Cluster Images ---
    print(f"Generating Super Image with Representative Cluster Images for {algo_name}...")

    # Group image paths by cluster ID, excluding noise for representative images
    cluster_image_map = {i: [] for i in range(num_clusters_found)}  # Only positive cluster IDs
    for i, cluster_id in enumerate(clusters):
        if cluster_id != -1:  # Exclude noise for representative images
            cluster_image_map[cluster_id].append(all_image_paths[i])

    representative_images_pil = []

    # Sort cluster IDs to ensure consistent super image order
    positive_cluster_ids = sorted([c_id for c_id in cluster_image_map.keys() if c_id >= 0])

    for cluster_id in positive_cluster_ids:
        if cluster_image_map[cluster_id]:
            random_image_path = random.choice(cluster_image_map[cluster_id])
            img_pil = Image.open(random_image_path).convert('RGB')
            img_pil = img_pil.resize(REP_IMAGE_DISPLAY_SIZE)
            img_pil = add_text_to_image(img_pil, f"C: {cluster_id}")
            representative_images_pil.append(img_pil)
        else:
            # Create a placeholder for empty clusters (if any positive cluster is empty)
            print(f"  Warning: Cluster {cluster_id} is empty or has no images. Adding a placeholder.")
            placeholder_img = Image.new('RGB', REP_IMAGE_DISPLAY_SIZE, color='lightgray')
            placeholder_img = add_text_to_image(placeholder_img, f"C: {cluster_id}\n(Empty)")
            representative_images_pil.append(placeholder_img)

    if representative_images_pil:
        # Combine images into a super image (horizontal strip)
        # Calculate optimal grid layout (e.g., 5 images per row)
        images_per_row = min(5, len(representative_images_pil))
        num_rows = (len(representative_images_pil) + images_per_row - 1) // images_per_row

        row_widths = []
        row_heights = []

        current_img_idx = 0
        for r in range(num_rows):
            row_imgs = representative_images_pil[current_img_idx: current_img_idx + images_per_row]
            row_widths.append(sum(img.width for img in row_imgs) + 10 * (len(row_imgs) - 1))
            row_heights.append(max(img.height for img in row_imgs))
            current_img_idx += images_per_row

        total_width = max(row_widths) if row_widths else 0
        total_height = sum(row_heights) + 10 * (num_rows - 1) if row_heights else 0

        if total_width == 0 or total_height == 0:
            print("  No images to form a super image.")
            return  # Exit if no images

        super_image = Image.new('RGB', (total_width, total_height), color='white')

        y_offset = 0
        current_img_idx = 0
        for r in range(num_rows):
            x_offset = 0
            row_imgs = representative_images_pil[current_img_idx: current_img_idx + images_per_row]
            for img in row_imgs:
                super_image.paste(img, (x_offset, y_offset))
                x_offset += img.width + 10  # Add padding between images
            y_offset += row_heights[r] + 10  # Add padding between rows
            current_img_idx += images_per_row

        super_image_filename = get_timestamp_filename("cluster_representatives", algo_name.lower().replace(" ", "_"),
                                                      "png")
        super_image.save(super_image_filename)
        print(f"  Cluster representative super image saved as '{super_image_filename}'")
    else:
        print("  No representative images to combine (perhaps all points are noise or no clusters formed).")


# --- Main Clustering Logic ---

# --- K-Means Clustering ---
print("\n--- Performing K-Means Clustering ---")
inertia = []
for k in K_RANGE:
    # n_init='auto' is the default in scikit-learn 1.4+, n_init=10 for older versions
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto' if hasattr(KMeans, 'n_init') else 10)
    kmeans.fit(embeddings)
    inertia.append(kmeans.inertia_)
    # print(f"K={k}, Inertia: {kmeans.inertia_:.2f}")

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(K_RANGE, inertia, marker='o')
plt.title('Elbow Method for Optimal K (K-Means)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.xticks(list(K_RANGE))
plt.grid(True)

elbow_plot_filename = get_timestamp_filename("elbow_method_plot", "kmeans", "png")
plt.savefig(elbow_plot_filename)
print(f"Elbow method plot saved as '{elbow_plot_filename}'")
plt.close()  # Close plot to free memory

# Simple heuristic to suggest optimal K
diffs = np.diff(inertia)
diffs_of_diffs = np.diff(diffs)

optimal_k_suggested = K_RANGE[0]  # Default to smallest
if len(diffs_of_diffs) > 0:
    optimal_k_index_in_diffs_of_diffs = np.argmin(diffs_of_diffs)
    optimal_k_suggested = K_RANGE[optimal_k_index_in_diffs_of_diffs + 1]
    print(f"\nSuggested Optimal K for K-Means (Elbow Method Heuristic): {optimal_k_suggested}")
else:
    print("\nNot enough data points for robust elbow heuristic for K-Means. Defaulting to smallest K.")

# Perform K-Means with optimal K
kmeans_optimal = KMeans(n_clusters=optimal_k_suggested, random_state=42,
                        n_init='auto' if hasattr(KMeans, 'n_init') else 10)
kmeans_clusters = kmeans_optimal.fit_predict(embeddings)
process_clustering_results(embeddings, all_image_paths, kmeans_clusters, "K-Means", optimal_k_suggested)

# --- DBSCAN Clustering ---
print("\n--- Performing DBSCAN Clustering ---")
# DBSCAN does not require 'n_clusters' but needs 'eps' and 'min_samples'
# These parameters are highly dependent on the dataset and need tuning.
# We use default values here for demonstration.
dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
dbscan_clusters = dbscan.fit_predict(embeddings)

# Count number of unique clusters found by DBSCAN (excluding noise if present)
n_clusters_dbscan = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
print(f"DBSCAN found {n_clusters_dbscan} clusters (plus noise points if any).")
process_clustering_results(embeddings, all_image_paths, dbscan_clusters, "DBSCAN", n_clusters_dbscan)

# --- Hierarchical Clustering (AgglomerativeClustering) ---
print("\n--- Performing Hierarchical Clustering (AgglomerativeClustering) ---")
# For hierarchical clustering, we use the optimal K found by K-Means for consistency
# Or a default if K-Means optimal K wasn't robustly determined.
n_clusters_hierarchical = optimal_k_suggested if optimal_k_suggested > 1 else 3  # Ensure at least 2 clusters for HC

# n_init is not applicable to AgglomerativeClustering
#hierarchical = AgglomerativeClustering(n_clusters=n_clusters_hierarchical)
hierarchical = AgglomerativeClustering()
hierarchical_clusters = hierarchical.fit_predict(embeddings)
process_clustering_results(embeddings, all_image_paths, hierarchical_clusters, "Hierarchical Clustering",
                           n_clusters_hierarchical)

print("\nScript finished successfully! Check the 'results' folder for output files.")
