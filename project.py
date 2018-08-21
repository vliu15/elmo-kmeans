import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def pca(params):
    '''
    Computes PCA to reduce dimensionality.
    :params.embedding_file: embeddings to reduce
    :params.pca_file: output file for reduced embeddings
    '''

    vectors = np.load(params.embedding_file)

    pc = PCA(n_components=params_pc_n_components)

    vectors_pc = pc.fit_transform(vectors)
    np.save(params.pca_file, vectors_pc)

def tsne(params):
    '''
    Computes t-SNE to reduce dimensionality.
    :params.embedding_file: embeddings to reduce
    :params.tsne_file: output file for reduced embeddings
    '''

    vectors = np.load(params.embedding_file)

    ts = TSNE(n_components=params.ts_n_components,
              perplexity=params.ts_perplexity,
              learning_rate=params.ts_learning_rate,
              n_iter=params.ts_n_iter)

    vectors_ts = ts.fit_transform(vectors)
    np.save(params.tsne_file, vectors_ts)
