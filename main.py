import numpy as np

from util import embed
from sif import sif
from cluster import kmeans, opt_k, hierarch_k
from project import pca, tsne
from meta import write_meta
from tensorboard import tensorboard

import argparse, os

# ELMo model to create embeddings
elmo_options_file = os.path.join(os.getcwd(), "model", "options.json")
elmo_weights_file = os.path.join(os.getcwd(), "model", "weights.hdf5")

# GloVe model to create embeddings
glove_word_file = os.path.join(os.getcwd(), "model", "glove.840B.300d.txt")
# glove_char_file = os.path.join(os.getcwd(), "model", "glove.840B.300d-char.txt")

# output files
filename = "medica-s.txt"
sentence_dir = os.path.join(os.getcwd(), "data")
# sentence_dir = os.getcwd()
sentence_file = os.path.join(sentence_dir, filename)
if sentence_file == os.path.join(sentence_dir, ""):
    print("Specify sentence file.")
    raise NameError

output_dir = os.path.join(os.getcwd(), "rapids", os.path.splitext(filename)[0])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

embedding_file = os.path.join(output_dir, "embeddings.npy")
sif_file = os.path.join(output_dir, "embeddings_sif.npy")
pca_file = os.path.join(output_dir, "embeddings_pc.npy")
tsne_file = os.path.join(output_dir, "embeddings_ts.npy")
km_labels_file = os.path.join(output_dir, "km_labels.json")
km_opt_file = os.path.join(output_dir, "km_opt.csv")
metadata_file = os.path.join(output_dir, "metadata.tsv")

# parse arguments at runtime
parser = argparse.ArgumentParser()

# files
parser.add_argument("--sentence_file", nargs='?', default=sentence_file, type=str, help="file for embedding")
parser.add_argument("--embedding_file", nargs='?', default=embedding_file, type=str, help="file for sentence embeddings")
parser.add_argument("--sif_file", nargs='?', default=sif_file, type=str, help="file for SIF embeddings")
parser.add_argument("--tsne_file", nargs='?', default=tsne_file, type=str, help="file for t-SNE embeddings")
parser.add_argument("--km_labels_file", nargs='?', default=km_labels_file, type=str, help="file for KMeans cluster labels")
parser.add_argument("--km_opt_file", nargs='?', default=km_opt_file, type=str, help="file for KMeans cluster-inertia-silhouette output")
parser.add_argument("--metadata_file", nargs='?', default=metadata_file, type=str, help="file for TensorBoard metadata")

# mode
parser.add_argument("--mode", nargs='?', default="embed", type=str, help="embed, sif, cluster, project, metadata, tensorboard")

# embed
parser.add_argument("--encoding", nargs='?', default="ascii", type=str, help="encoding of input sentence file")
parser.add_argument("--errors", nargs='?', default="ignore", type=str, help="error-handling of reading encoded input")

parser.add_argument("--elmo", nargs='?', default=True, type=bool, help="use ELMo for embeddings")
parser.add_argument("--elmo_options_file", nargs='?', default=elmo_options_file, type=str, help="options file for ELMo embedding")
parser.add_argument("--elmo_weights_file", nargs='?', default=elmo_weights_file, type=str, help="weights file for ELMo embedding")
parser.add_argument("--elmo_cuda_device", nargs='?', default=0, type=int, help="GPU device to run on")

parser.add_argument("--glove", nargs='?', default=False, type=bool, help="use GloVe for embeddings")
parser.add_argument("--glove_word_file", nargs='?', default=glove_word_file, type=str, help="word file for GloVe embedding")
# parser.add_argument("--glove_char_file", nargs='?', default=glove_char_file, type=str, help="char file for GloVe embedding")

parser.add_argument("--bilm_layer_index", nargs='?', default=2, type=int, help="which bilm layer of ELMo to use, indexed from 0 (-1 for average)")
parser.add_argument("--sum_word_vecs", nargs='?', default=False, type=bool, help="sum word vectors in the same sentence")
parser.add_argument("--avg_word_vecs", nargs='?', default=True, type=bool, help="average word vectors in same sentence")
parser.add_argument("--concat_word_vecs", nargs='?', default=False, type=bool, help="concatenate word vectors in the same sentence")
parser.add_argument("--max_pool_word_vecs", nargs='?', default=False, type=bool, help="max pooling across word vectors in the same sentence")
parser.add_argument("--max_transcript_len", nargs='?', default=30, type=int, help="if concatenating, length to pad/truncate to")

# sif
parser.add_argument("--sif_rmpc", nargs='?', default=1, type=int, help="number of principal components to remove")

# cluster
parser.add_argument("--kmeans", nargs='?', default=False, type=bool, help="use KMeans clustering")
parser.add_argument("--kmeans_dir", nargs='?', default=os.path.join(output_dir, "kmeans"), type=str, help="output file for KMeans clusters")
parser.add_argument("--n_clusters", nargs='?', default=10, type=int, help="n_clusters in KMeans function")
parser.add_argument("--n_init", nargs='?', default=10, type=int, help="n_init in KMeans function")
parser.add_argument("--max_iter", nargs='?', default=300, type=int, help="max_iter in KMeans function")
parser.add_argument("--verbose", nargs='?', default=False, type=bool, help="verbose in KMeans function")
parser.add_argument("--n_jobs", nargs='?', default=-1, type=int, help="n_jobs in KMeans function")
parser.add_argument("--algorithm", nargs='?', default="auto", type=str, help="algorithm in KMeans function")

parser.add_argument("--opt_k", nargs='?', default=True, type=bool, help="find optimal k")
parser.add_argument("--min_k", nargs='?', default=10, type=int, help="minimum k to try")
parser.add_argument("--max_k", nargs='?', default=110, type=int, help="maximum k to try")
parser.add_argument("--n_k", nargs='?', default=10, type=int, help="number of k's to try")

parser.add_argument("--hierarch_k", nargs='?', default=False, type=bool, help="compute kmeans hierarchically")
parser.add_argument("--hierarch_dir", nargs='?', default=os.path.join(output_dir, "hierarchy"), type=str, help="directory for hierarchy clusters")
parser.add_argument("--split_size", nargs='?', default=2, type=int, help="number of clusters at each level")
parser.add_argument("--n_iter", nargs='?', default=4, type=int, help="number of levels of hierarchy")

# project
parser.add_argument("--pca", nargs='?', default=False, type=bool, help="use pca for visualization")
parser.add_argument("--pc_n_components", nargs='?', default=3, type=int, help="n_components in PCA function")

parser.add_argument("--tsne", nargs='?', default=True, type=bool, help="use tsne for visualization")
parser.add_argument("--ts_n_components", nargs='?', default=3, type=int, help="n_components in TSNE function")
parser.add_argument("--ts_perplexity", nargs='?', default=50, help="perplexity n TSNE function")
parser.add_argument("--ts_learning_rate", nargs='?', default=10, type=int, help="learning_rate in TSNE function")
parser.add_argument("--ts_n_iter", nargs='?', default=5000, type=int, help="n_iter in TSNE function")

# metadata
parser.add_argument("--meta_labels", nargs='?', default=True, type=bool, help="use labels in metadata")
parser.add_argument("--meta_labels_file", nargs='?', default=km_labels_file, type=str, help="labels file to be used in metadata")

# tensorboard
parser.add_argument("--log_dir", nargs='?', default=os.path.join(output_dir, "tensorboard"), type=str, help="log directory for TensorBoard")

params = parser.parse_args()


if __name__ == "__main__":

    if params.mode == "embed":
        embed(params)

    if params.mode == "sif":
        sif(params)

    if params.mode == "cluster":
        if params.kmeans:
            kmeans(params)
        if params.opt_k:
            opt_k(params)
        if params.hierarch_k:
            hierarch_k(params)

    if params.mode == "project":
        if params.pca:
            pca(params)
        if params.tsne:
            tsne(params)

    if params.mode == "metadata":
        # optional write cluster labels to metadata
        if not params.meta_labels:
            params.meta_labels_file = None
        write_meta(params)

    if params.mode == "tensorboard":
        tensorboard(params)
