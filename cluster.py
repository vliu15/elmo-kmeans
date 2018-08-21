import numpy as np
import json
import math

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from tqdm import *


def dbscan(params):
    '''
    Cluster arrays using DBSCAN.
    :params.embedding_file: 2D NumPy arrays
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
    :params.embedding_file: 2D NumPy arrays
    '''

    vectors = np.load(params.embedding_file)

    if params.km_opt:
        opt = []
        incr = int((params.max_k - params.min_k) / params.n_k)

        # loop through different k values
        for i in tqdm(range(params.n_k + 1)):
            cur_n = params.max_k - i * incr
            km = KMeans(n_clusters=cur_n,
                        n_init=params.km_n_init,
                        max_iter=params.km_max_iter,
                        verbose=params.km_verbose,
                        n_jobs=params.km_n_jobs,
                        algorithm=params.km_algorithm).fit(vectors)
            labels = km.labels_
            inertia = km.inertia_
            silhouette = silhouette_score(vectors, labels)
            opt.append(str(cur_n) + '\t' + str(inertia) + '\t' + str(silhouette) + '\n')
            print("Clusters: %d \t Inertia: %.5f \t Silhouette: %.5f \n" % (cur_n, inertia, silhouette))

        with open(params.km_opt_file, 'w') as f:
            f.writelines(opt)

    else:
        km = KMeans(n_clusters=params.km_n_clusters,
                    n_init=params.km_n_init,
                    max_iter=params.km_max_iter,
                    verbose=params.km_verbose,
                    n_jobs=params.km_n_jobs,
                    algorithm=params.km_algorithm).fit(vectors)
        labels = km.labels_
        inertia = km.inertia_
        silhouette = silhouette_score(vectors, labels)
        print("Clusters: %d \t Inertia: %.5f \t 'silhouette: %.5f\n" % (params.km_n_clusters, inertia, silhouette))

        # write list of labels to output
        with open(params.km_labels_file, 'w') as f:
            json.dump(labels.tolist(), f)
