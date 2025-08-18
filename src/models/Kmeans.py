import torch

def kmeans(X, K, num_iters=100, device="cpu"):
    """
    PyTorch ile iteratif KMeans
    X: (N, D) tensörü
    K: küme sayısı
    num_iters: maksimum iterasyon
    device: "cuda" veya "cpu"
    """
    X = X.to(device)
    N, D = X.shape
    
    
    indices = torch.randperm(N)[:K]
    centroids = X[indices]

    for i in range(num_iters):
        
        distances = torch.cdist(X, centroids)

        
        labels = torch.argmin(distances, dim=1)

       
        new_centroids = torch.stack([
            X[labels == k].mean(dim=0) if torch.any(labels == k) else centroids[k]
            for k in range(K)
        ])

        
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids

    return centroids, labels


