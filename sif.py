'''
Taken from https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py
Place in pipeline: to better existing sentence embeddings to be used for post analysis
'''

import numpy as np
from sklearn.decomposition import TruncatedSVD

def compute_pc(vectors,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param vectors: vectors[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """

    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(vectors)
    return svd.components_

def remove_pc(vectors, npc=1):
    """
    Remove the projection on the principal components
    :param vectors: vectors[i,:] is a data point
    :param npc: number of principal components to remove
    :return: vectors_pc[i, :] is the data point after removing its projection
    """

    pc = compute_pc(vectors, npc)
    if npc==1:
        vectors_pc = vectors - vectors.dot(pc.transpose()) * pc
    else:
        vectors_pc = vectors - vectors.dot(pc.transpose()).dot(pc)
    return vectors_pc


def sif(params):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param params.embedding_file: numpy array of sentence embeddings
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    """

    vectors = np.load(params.embedding_file)
    if  params.sif_rmpc > 0:
        vectors_pc = remove_pc(vectors, params.sif_rmpc)

    np.save(params.sif_file, vectors_pc)
