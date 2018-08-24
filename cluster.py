import numpy as np
import json
import math

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import *


def kmeans(params):
    '''
    Cluster arrays using KMeans.
    :params.embedding_file: 2D NumPy arrays
    '''

    vectors = np.load(params.embedding_file)

    km = KMeans(n_clusters=params.n_clusters,
                n_init=params.n_init,
                max_iter=params.max_iter,
                verbose=params.verbose,
                n_jobs=params.n_jobs,
                algorithm=params.algorithm).fit(vectors)
    labels = km.labels_
    inertia = km.inertia_
    silhouette = silhouette_score(vectors, labels)
    print("Clusters: %d \t Inertia: %.5f \t 'silhouette: %.5f\n" % (params.n_clusters, inertia, silhouette))

    # write list of labels to output
    with open(params.km_labels_file, 'w') as f:
        json.dump(labels.tolist(), f)

def opt_k(params):
    '''
    Returns inertia and silhouette scores of k clusters in specified range.
    :params.embedding_file: 2D NumPy arrays
    :params.min_k: minimum k value to start
    :params.max_k: maximum k value to start
    :params.n_k: number of k values to try within range
    '''

    vectors = np.load(params.embedding_file)

    opt = []
    incr = int((params.max_k - params.min_k) / params.n_k)

    # loop through different k values
    for i in tqdm(range(params.n_k + 1)):
        cur_n = params.max_k - i * incr
        km = KMeans(n_clusters=cur_n,
                    n_init=params.n_init,
                    max_iter=params.max_iter,
                    verbose=params.verbose,
                    n_jobs=params.n_jobs,
                    algorithm=params.algorithm).fit(vectors)
        labels = km.labels_
        inertia = km.inertia_
        silhouette = silhouette_score(vectors, labels)
        opt.append(str(cur_n) + '\t' + str(inertia) + '\t' + str(silhouette) + '\n')
        print("Clusters: %d \t Inertia: %.5f \t Silhouette: %.5f \n" % (cur_n, inertia, silhouette))

    with open(params.km_opt_file, 'w') as f:
        f.writelines(opt)


def hierarch_k(params):

    vectors = np.load(params.embedding_file)
