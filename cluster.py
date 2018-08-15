import numpy as np
import json
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN, KMeans


def meanshift(params):
    '''
    Cluster arrays using MeanShift.
    :params params.embedding_file: 2D NumPy arrays
    '''

    vectors = np.load(params.embedding_file)

    # bandwidth = estimate_bandwidth(vectors)
    ms = MeanShift(bin_seeding=params.ms_bin_seeding)

    labels = ms.fit_predict(vectors)
    print("Number of estimated clusters using MeanShift: %d\n" % len(clusters))
    print("Noise: %d\n" %  len([i for i in labels if i == -1]))

    # write list of labels to output
    with open(params.ms_labels_file, 'w') as f:
        json.dump(labels.tolist(), f)


def dbscan(params):
    '''
    Cluster arrays using DBSCAN.
    :params params.embedding_file: 2D NumPy arrays
    '''

    vectors = np.load(params.embedding_file)

    db = DBSCAN(eps=params.db_eps,
                min_samples=params.db_min_samples,
                metric=params.db_metric,
                algorithm=params.db_algorithm,
                n_jobs=params.db_n_jobs)

    labels = db.fit_predict(vectors)
    print("Number of estimated clusters using DBSCAN: %d\n" % (len(set(labels)) - (1 if -1 in labels else 0)))
    print("Noise: %d\n" %  len([i for i in labels if i == -1]))

    # write list of labels to output
    with open(params.db_labels_file, 'w') as f:
        json.dump(labels.tolist(), f)


def optics(params):
    '''
    Cluster arrays using OPTICS.
    :params params.embedding_file: 2D NumPy arrays
    '''

    vectors = np.load(params.embedding_file)

    op = OPTICS(min_samples=params.op_min_samples)
    labels = op.fit_predict(vectors)
    print("Number of estimated clusters using OPTICS: %d\n" % (len(set(labels)) - (1 if -1 in labels else 0)))
    print("Noise: %d\n" % len([i for i in labels if i == -1]))

    # write list of labels to output
    with open(params.op_labels_file, 'w') as f:
        json.dump(labels.tolist(), f)


def kmeans(params):
    '''
    Cluster arrays using KMeans.
    :params params.embedding_file: 2D NumPy arrays
    '''

    vectors = np.load(params.embedding_file)

    km = KMeans(n_clusters=km_n_clusters,
                n_init=km_n_init,
                max_iter=km_max_iter,
                n_jobs=km_n_jobs,
                algorithm=km_algorithm)

    labels = km.fit_predict(vectors)
    print("Number of estimated clusters using KMeans: %d\n" % (len(set(labels)) - (1 if -1 in labels else 0)))
    print("Noise: %d\n" % len([i for i in labels if i == -1]))

    # write list of labels to output
    with open(params.km_labels_file, 'w') as f:
        json.dump(labels.tolist(), f)
