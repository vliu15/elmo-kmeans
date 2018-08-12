import numpy as np
import json
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from sklearn.manifold import TSNE

# cluster NumPy arrays of sentence vectors using MeanShift
def meanshift(params vectors_np):

    # compute clusters
    bandwidth = estimate_bandwidth(vectors_np)

    ms = MeanShift(bin_seeding=params.ms_bin_seeding)
    labels = ms.fit_predict(vectors_np)
    clusters = ms.cluster_centers_
    print("Number of estimated clusters using MeanShift: %d\n" % len(clusters))
    print("Noise: %d\n" %  len([i for i in labels if i == -1]))

    # write list of labels to output
    with open(params.ms_labels_file, 'w') as f:
        json.dump(labels.tolist(), f)


# cluster NumPy arrays of sentence vectors using DBSCAN
def dbscan(params, vectors_np):
  
    # compute clusters
    db = DBSCAN(eps=params.db_eps,
                min_samples=params.db_min_samples,
                metric=params.db_metric,
                algorithm=params.db_algorithm,
                n_jobs=params.db_n_jobs)

    labels = db.fit_predict(vectors_np)
    print("Number of estimated clusters using DBSCAN: %d\n" % (len(set(labels)) - (1 if -1 in labels else 0)))
    print("Noise: %d\n" %  len([i for i in labels if i == -1]))

    # write list of labels to output
    with open(params.db_labels_file, 'w') as f:
        json.dump(labels.tolist(), f)


# cluster NumPy arrays of sentence vectors using OPTICS
def optics(params, vectors_np):

    # compute clusters
    op = OPTICS(min_samples=params.op_min_samples)
    labels = op.fit_predict(vectors_np)
    ids = op.core_sample_indices_
    print("Number of estimated clusters using OPTICS: %d\n" % (len(set(labels)) - (1 if -1 in labels else 0)))
    print("Noise: %d\n" % len([i for i in labels if i == -1]))

    # write list of labels to output
    with open(params.op_labels_file, 'w') as f:
        json.dump(labels.tolist(), f)


# project NumPy arrays of sentence vectors into 3D using TSNE
def tsne(params, vectors_np):

    # use t-SNE
    ts = TSNE(n_components=params.ts_n_components,
              perplexity=params.ts_perplexity,
              learning_rate=params.ts_learning_rate,
              n_iter=params.ts_n_iter)

    vectors_ts = ts.fit_transform(vectors_np)

    np.save(params.ts_embed_file, vectors_ts)
    return vectors_ts
