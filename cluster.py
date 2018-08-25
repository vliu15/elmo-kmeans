import numpy as np
import shutil
import json
import math
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import *


def kmeans(params):
    '''
    Cluster arrays using KMeans.
    :params.embedding_file: 2D NumPy arrays
    '''

    vectors = np.load(params.embedding_file)
    sentences = open(params.sentence_file, 'r').read().splitlines()

    # clear current directory of groups
    if os.path.exists(params.kmeans_dir):
        shutil.rmtree(params.kmeans_dir)
    os.makedirs(params.kmeans_dir)

    # compute KMeans
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
    
    # write labels to .txt files
    for i in tqdm(range(len(labels))):
        label = labels[i]
        file_name = str(label) + '.txt'
        with open(os.path.join(params.kmeans_dir, file_name), 'a') as f:
            f.write(sentences[i] + '\n')


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
    '''
    Computes KMeans hierarchically.
    :params.embedding_file: 2D Numpy Arrays
    :params.snetence_file: file of sentences
    :params.n_iter: number of hierarchy levels
    :params.split_size: number of clusters per level
    '''

    vectors = np.load(params.embedding_file)
    sentences = open(params.sentence_file, 'r').read().splitlines()

    # clear current directory of groups
    if os.path.exists(params.hierarch_dir):
        shutil.rmtree(params.hierarch_dir)
    os.makedirs(params.hierarch_dir)

    # cluster iteratively
    iter_k(params, vectors, sentences, params.n_iter, '')


def iter_k(params, vectors, sentences, i, filename):
    '''
    Helper function for hierarch_k to perform iterations.
    :vectors: NumPy arrays to be clustered
    :sentences: list of sentences corresponding to vectors
    :i: ith iteration in hierarchy
    :filename: file to write 
    '''

    if i == 0:
        return

    km = KMeans(n_clusters=2,
                n_init=params.n_init,
                max_iter=params.max_iter,
                verbose=params.verbose,
                n_jobs=params.n_jobs,
                algorithm=params.algorithm).fit(vectors)
    labels = km.labels_
    for j in range(params.split_size):
        cluster_s, cluster_v = [], []
        for k in range(len(labels)):
            if labels[k] == j:
                cluster_s.append(sentences[k])
                cluster_v.append(vectors[k])
        if len(cluster_s) >= params.n_init:
            iter_k(params, np.stack(cluster_v), cluster_s, i-1, filename + str(j))
        with open(os.path.join(params.hierarch_dir, filename + str(j)), 'w') as f:
            f.writelines([s + '\n' for s in cluster_s])

