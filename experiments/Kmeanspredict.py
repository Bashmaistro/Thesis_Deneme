import numpy as np
import cupy as cp
from cuml.cluster import KMeans as cuKMeans
import gc



def sample_from_clusters(data_array: np.ndarray,
                         saved_centroids: np.ndarray,
                         n_clusters: int,
                         target_clusters: list,
                         total_slices: int,
                         batch_size: int = 128,
                         target_shape=(240, 240)) -> np.ndarray:

    data = data_array.reshape(data_array.shape[0], -1)

    centroids_cp = cp.asarray(saved_centroids)
    kmeans_model = cuKMeans(n_clusters=n_clusters, init=centroids_cp, max_iter=100)

    all_labels = []
    cluster_to_samples = {cluster: [] for cluster in target_clusters}

    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i + batch_size]
        batch_cp = cp.asarray(batch.astype(np.float32))
        labels_cp = kmeans_model.predict(batch_cp)
        labels_np = cp.asnumpy(labels_cp)

        all_labels.extend(labels_np.tolist())

        for j, label in enumerate(labels_np):
            if label in cluster_to_samples:
                cluster_to_samples[label].append(batch[j])

    all_labels_np = np.array(all_labels)
    total_cluster_counts = {c: (all_labels_np == c).sum() for c in target_clusters}
    total_count_in_targets = sum(total_cluster_counts.values())

    cluster_slice_targets = {}
    final_samples = []

    print("\n--- Cluster Özet ---")
    for c in target_clusters:
        cluster_count = total_cluster_counts[c]
        cluster_ratio = cluster_count / total_count_in_targets if total_count_in_targets > 0 else 0
        target_count = int(round(total_slices * cluster_ratio))
        cluster_slice_targets[c] = target_count

        available_count = len(cluster_to_samples[c])
        used_count = min(available_count, target_count)

        print(f"Cluster {c} | Veri içindeki oran: {cluster_ratio:.2%} | "
              f"Toplam: {cluster_count} | Hedef: {target_count} | Alınan: {used_count}")

        final_samples.extend(cluster_to_samples[c][:used_count])

    remaining_slices = total_slices - len(final_samples)

    if remaining_slices > 0:
        print(f"Eksik kalan {remaining_slices} örneği diğer kümelerden ekliyoruz.")
        all_labels_np = np.array(all_labels)
        other_clusters = [i for i in range(n_clusters) if i not in target_clusters]

        cluster_to_samples_other = {c: [] for c in other_clusters}
        for i, label in enumerate(all_labels_np):
            if label in other_clusters:
                cluster_to_samples_other[label].append(data[i])

        for c in other_clusters:
            if remaining_slices <= 0:
                break
            available = len(cluster_to_samples_other[c])
            if available > 0:
                needed = min(remaining_slices, available)
                final_samples.extend(cluster_to_samples_other[c][:needed])
                remaining_slices -= needed

    # Hedef şekil: Her örnek 240x240 olarak yeniden boyutlandırılıyor
    collected_array = np.array(final_samples)
    collected_array_resized = np.array([np.resize(sample, target_shape) for sample in collected_array])

    if collected_array_resized.shape[0] < total_slices:
        print(f"⚠️ Uyarı: Hedeflenen {total_slices} slice sayısına ulaşılamadı. Toplanan: {collected_array_resized.shape[0]}")

    print(f"\nToplam hedef: {total_slices}")
    print(f"Toplam toplanan: {collected_array_resized.shape[0]}")
    print("----------------------\n")

    # Bellek temizliği
    del all_labels_np
    del cluster_to_samples
    del all_labels
    del final_samples
    gc.collect()

    return collected_array_resized