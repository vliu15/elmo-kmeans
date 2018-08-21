import tensorflow as tf
import numpy as np

from util import embed, tokenize
from sif import sif
from project import pca, tsne
from cluster import dbscan, kmeans
from meta import write_meta
from tensorboard import tensorboard
from analyze import write_groups, remove_groups

import argparse, os

# ELMo model to create embeddings
elmo_options_file = os.path.join(os.getcwd(), "model", "options.json")
elmo_weights_file = os.path.join(os.getcwd(), "model", "weights.hdf5")

# GloVe model to create embeddings
glove_word_file = os.path.join(os.getcwd(), "model", "glove.840B.300d.txt")
glove_char_file = os.path.join(os.getcwd(), "model", "glove.840B.300d-char.txt")

# output files
output_dir = os.path.join(os.getcwd(), "output", "l3-avg", "trimmed")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sentence_file = os.path.join(output_dir, "sentences.txt")
# embedding_file = os.path.join(output_dir, "embeddings.npy")
embedding_file = os.path.join(output_dir, "embeddings_sif.npy")
sif_file = os.path.join(output_dir, "embeddings_sif.npy")
pca_file = os.path.join(output_dir, "embeddings_pc.npy")
tsne_file = os.path.join(output_dir, "embeddings_ts.npy")
db_labels_file = os.path.join(output_dir, "db_labels.json")
km_labels_file = os.path.join(output_dir, "km_labels-25.json")
km_opt_file = os.path.join(output_dir, "km_opt.csv")
metadata_file = os.path.join(output_dir, "metadata.tsv")

# flags for adjusting parameters at runtime
flags = tf.flags

# files
flags.DEFINE_string("sentence_file", sentence_file, "file for embedding")
flags.DEFINE_string("embedding_file", embedding_file, "file for sentence embeddings")
flags.DEFINE_string("sif_file", sif_file, "file for SIF embeddings")
flags.DEFINE_string("tsne_file", tsne_file, "file for t-SNE embeddings")
flags.DEFINE_string("db_labels_file", db_labels_file, "file for DBSCAN cluster labels")
flags.DEFINE_string("km_labels_file", km_labels_file, "file for KMeans cluster labels")
flags.DEFINE_string("km_opt_file", km_opt_file, "file for KMeans cluster-inertia-silhouette output")
flags.DEFINE_string("metadata_file", metadata_file, "file for TensorBoard metadata")

# mode
flags.DEFINE_string("mode", "embed", "embed, sif, project, cluster, metadata, tensorboard, analyze")

# embed
flags.DEFINE_boolean("elmo", True, "use ELMo for embeddings")
flags.DEFINE_string("elmo_options_file", elmo_options_file, "options file for ELMo embedding")
flags.DEFINE_string("elmo_weights_file", elmo_weights_file, "weights file for ELMo embedding")

flags.DEFINE_boolean("glove", False, "use GloVe for embeddings")
flags.DEFINE_string("glove_word_file", glove_word_file, "word file for GloVe embedding")
flags.DEFINE_string("glove_char_file", glove_char_file, "char file for GloVe embedding")

flags.DEFINE_integer("bilm_layer_index", 2, "which bilm layer of ELMo to use, indexed from 0 (-1 for average)")
flags.DEFINE_boolean("sum_word_vecs", False, "sum word vectors in the same sentence")
flags.DEFINE_boolean("avg_word_vecs", True, "average word vectors in same sentence")
flags.DEFINE_boolean("concat_word_vecs", False, "concatenate word vectors in the same sentence")
flags.DEFINE_integer("max_transcript_len", 30, "if concatenating, length to pad/truncate to") # trans/sent, 75/30

# sif
flags.DEFINE_integer("sif_rmpc", 1, "number of principal components to remove")

# project
flags.DEFINE_boolean("pca", False, "use pca for visualization")
flags.DEFINE_integer("pc_n_components", 3, "n_components in PCA function")

flags.DEFINE_boolean("tsne", True, "us tsne for visualization")
flags.DEFINE_integer("ts_n_components", 3, "n_components in TSNE function")
flags.DEFINE_integer("ts_perplexity", 50, "perplexity n TSNE function")
flags.DEFINE_integer("ts_learning_rate", 10, "learning_rate in TSNE function")
flags.DEFINE_integer("ts_n_iter", 5000," n_iter in TSNE function")

# cluster
flags.DEFINE_boolean("dbscan", False, "use DBSCAN clustering")
flags.DEFINE_float("db_eps", 1, "eps in DBSCAN function")
flags.DEFINE_integer("db_min_samples", 10, "min_samples in DBSCAN function")
flags.DEFINE_string("db_metric", "euclidean", "metric in DBSCAN function")
flags.DEFINE_string("db_algorithm", "brute", "algorithm in DBSCAN function")
flags.DEFINE_integer("db_n_jobs", -1, "n_jobs in DBSCAN function")

flags.DEFINE_boolean("kmeans", True, "use KMeans clustering")
flags.DEFINE_integer("km_n_clusters", 25, "n_clusters in KMeans function")
flags.DEFINE_integer("km_n_init", 10, "n_init in KMeans function")
flags.DEFINE_integer("km_max_iter", 300, "max_iter in KMeans function")
flags.DEFINE_boolean("km_verbose", False, "verbose in KMeans function")
flags.DEFINE_integer("km_n_jobs", -1, "n_jobs in KMeans function")
flags.DEFINE_string("km_algorithm","auto", "algorithm in KMeans function")

flags.DEFINE_boolean("km_opt", False, "find optimal k")
flags.DEFINE_integer("min_k", 5, "minimum k to try")
flags.DEFINE_integer("max_k", 155, "maximum k to try")
flags.DEFINE_integer("n_k", 15, "number of k's to try")

# metadata
flags.DEFINE_boolean("meta_labels", True, "use labels in metadata")
flags.DEFINE_string("meta_labels_file", km_labels_file, "labels file to be used in metadata")

# tensorboard
flags.DEFINE_string("log_dir", os.path.join(output_dir, "tensorboard"), "log directory for TensorBoard")

# analyze
flags.DEFINE_string("group_labels_file", km_labels_file, "labels file to write sentence groups")
flags.DEFINE_boolean("write_groups", True, "write clustered sentences to files")
flags.DEFINE_string("group_dir", os.path.join(output_dir, "groups"), "directory for clustered sentences")
flags.DEFINE_boolean("remove_groups", False, "remove specificed clusters from sentences")
flags.DEFINE_string("trim_dir", os.path.join(output_dir, "trimmed"), "directory for trimmed version")

params = flags.FLAGS


if __name__ == "__main__":

    if params.mode == "embed":
        tokenized = tokenize(params)

        with open(params.sentence_file, 'r') as f:
            tokenized = f.read().splitlines()

        # set up TF with GPU usage
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            emb = sess.run(embed(params, tokenized))
            print("Saving embeddings...\n")
            np.save(params.embedding_file, emb)

    if params.mode == "sif":
        sif(params)

    if params.mode == "project":
        if params.pca:
            pca(params)
        if params.tsne:
            tsne(params)

    if params.mode == "cluster":
        if params.dbscan:
            dbscan(params)
        if params.kmeans:
            kmeans(params)

    if params.mode == "metadata":
        if not params.meta_labels:
            params.meta_labels_file = None
        write_meta(params)

    if params.mode == "tensorboard":
        tensorboard(params)

    if params.mode == "analyze":
        if params.write_groups:
            write_groups(params)
        if params.remove_groups:
            remove_groups(params)
