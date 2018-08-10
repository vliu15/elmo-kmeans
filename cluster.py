import numpy as np
import json
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from sklearn.manifold import TSNE

# cluster NumPy arrays of sentence vectors using MeanShift
def meanshift(vectors_np, out_file):

    # compute clusters
    bandwidth = estimate_bandwidth(vectors_np)

    ms = MeanShift(bin_seeding=True)
    labels = ms.fit_predict(vectors_np)
    clusters = ms.cluster_centers_
    print("Number of estimated clusters using MeanShift: %d\n" % len(clusters))
    print("Noise: %d\n" %  len([i for i in labels if i == -1]))

    # write list of labels to output
    with open(out_file, 'w') as f:
        json.dump(labels.tolist(), f)


# cluster NumPy arrays of sentence vectors using DBSCAN
def dbscan(vectors_np, out_file):
  
    # compute clusters
    db = DBSCAN(eps=10, min_samples=100, metric='euclidean', algorithm='brute', n_jobs=-1)
    labels = db.fit_predict(vectors_np)
    print("Number of estimated clusters using DBSCAN: %d\n" % (len(set(labels)) - (1 if -1 in labels else 0)))
    print("Noise: %d\n" %  len([i for i in labels if i == -1]))

    # write list of labels to output
    with open(out_file, 'w') as f:
        json.dump(labels.tolist(), f)


# cluster NumPy arrays of sentence vectors using OPTICS
def optics(vectors_np, out_file):

    # compute clusters
    op = OPTICS(min_samples=50, p=2)
    labels = op.fit_predict(vectors_np)
    ids = op.core_sample_indices_
    print("Number of estimated clusters using OPTICS: %d\n" % (len(set(labels)) - (1 if -1 in labels else 0)))
    print("Noise: %d\n" % len([i for i in labels if i == -1]))

    # write list of labels to output
    with open(out_file, 'w') as f:
        json.dump(labels.tolist(), f)


# project NumPy arrays of sentence vectors into 3D using TSNE
def tsne(vectors_np, out_file):
   ts = TSNE(n_components=3, perplexity=50, learning_rate=20, n_iter=5000)
   vectors_ts = ts.fit_transform(vectors_np)

   np.save(out_file, vectors_ts)
   return vectors_ts
