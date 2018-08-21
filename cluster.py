import numpy as np
import json
from sklearn.cluster import DBSCAN, KMeans


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


def kmeans(params):
    '''
    Cluster arrays using KMeans.
    :params params.embedding_file: 2D NumPy arrays
    '''

    vectors = np.load(params.embedding_file)

    km = KMeans(n_clusters=params.km_n_clusters,
                n_init=params.km_n_init,
                max_iter=params.km_max_iter,
                verbose=params.km_verbose,
                n_jobs=params.km_n_jobs,
                algorithm=params.km_algorithm).fit(vectors)

    labels = km.labels_
    inertia = km.inertia_
    print("Number of estimated clusters using KMeans: %d\n" % (len(set(labels)) - (1 if -1 in labels else 0)))
    print("Inertia: %.5f\n" % inertia)

    # write list of labels to output
    with open(params.km_labels_file, 'w') as f:
        json.dump(labels.tolist(), f)
