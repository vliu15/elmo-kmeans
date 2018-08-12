import tensorflow as tf
import numpy as np
import json
from allennlp.commands.elmo import ElmoEmbedder

from util import embed, tokenize
from cluster import meanshift, dbscan, optics, tsne
from meta import write_meta
from tensorboard import tensorboard

import argparse, os

# ELMo model to create embeddings
options_file = os.path.join(os.getcwd(), "model", "options.json")
weights_file = os.path.join(os.getcwd(), "model", "weights.hdf5")


# output files
output_dir = os.path.join(os.getcwd(), "output", "concat")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sentence_file = os.path.join(output_dir, "transcriptions.txt")
embedding_file = os.path.join(output_dir, "embeddings.npy")
ms_labels_file = os.path.join(output_dir, "ms_labels.json")
db_labels_file = os.path.join(output_dir, "db_labels.json")
op_labels_file = os.path.join(output_dir, "op_labels.json")
ts_embed_file = os.path.join(output_dir, "embeddings_ts.npy")
metadata_file = os.path.join(output_dir, "metadata.tsv")


# flags for adjusting parameters at runtime
flags = tf.flags

flags.DEFINE_string("mode", "embed", "embed, cluster, metadata, or tensorboard")

flags.DEFINE_integer("bilm_layer_index", 2, "which bilm layer of ELMo to use, indexed from 0 (-1 for average)")
flags.DEFINE_boolean("sum_word_vecs", False, "sum word vectors in the same sentence")
flags.DEFINE_boolean("concat_word_vecs", True, "concatenate word vectors in the same sentence")
flags.DEFINE_integer("max_transcript_len", 75, "if concatenating, length to pad/truncate to")

flags.DEFINE_boolean("use_tsne", False, "use t-SNE before clustering")
flags.DEFINE_integer("ts_n_components", 3, "n_components in TSNE function")
flags.DEFINE_integer("ts_perplexity", 50, "perplexity n TSNE function")
flags.DEFINE_integer("ts_learning_rate", 10, "learning_rate in TSNE function")
flags.DEFINE_integer("ts_n_iter", 5000," n_iter in TSNE function")

flags.DEFINE_boolean("use_ms", False, "use MeanShift clustering")
flags.DEFINE_boolean("ms_bin_seeding", True, "bin_seeding in MeanShift function")

flags.DEFINE_boolean("use_db", True, "use DBSCAN clustering")
flags.DEFINE_integer("db_eps", 0.15, "eps in DBSCAN function")
flags.DEFINE_integer("db_min_samples", 100, "min_samples in DBSCAN function")
flags.DEFINE_string("db_metric", "cosine", "metric in DBSCAN function")
flags.DEFINE_string("db_algorithm", "brute", "algorithm in DBSCAN function")
flags.DEFINE_integer("db_n_jobs", -1, "n_jobs in DBSCAN function")

flags.DEFINE_boolean("use_op", False, "use OPTICS clustering")
flags.DEFINE_integer("op_min_samples", 100, "min_samples in OPTICS function")

flags.DEFINE_boolean("meta_labels", False, "use labels in metadata")
flags.DEFINE_string("meta_labels_file", db_labels_file, "labels file to be used in metadata")

flags.DEFINE_string("log_dir", output_dir, "log directory for TensorBoard")

flags.DEFINE_string("sentence_file", sentence_file, "file for embedding")
flags.DEFINE_string("embedding_file", embedding_file, "file for sentence embeddings")
flags.DEFINE_string("ms_labels_file", ms_labels_file, "file for MeanShift cluster labels")
flags.DEFINE_string("db_labels_file", db_labels_file, "file for DBSCAN cluster labels")
flags.DEFINE_string("op_labels_file", op_labels_file, "file for OPTICS cluster labels")
flags.DEFINE_string("ts_embed_file", ts_embed_file, "file for t-SNE embeddings")
flags.DEFINE_string("metadata_file", metadata_file, "file for TensorBoard metadata")

params = flags.FLAGS

if __name__ == "__main__":

    if params.mode == "embed":

        # tokenize transcriptions
        tokenized = tokenize(params)

        # set up embedding with ELMo
        elmo = ElmoEmbedder(options_file, weights_file)

        # set up TF with GPU usage
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            emb = sess.run(embed(params, elmo, tokenized))
            np.save(params.embedding_file, emb)

    if params.mode == "cluster":

        # load embeddings.npy file
        vectors_np = np.load(params.embedding_file)

        # 3-d projection
        if params.use_tsne:
            vectors_np = tsne(params, vectors_np)

        # perform clustering
        if params.use_ms:
            mean_shift(params, vectors_np)
        if params.use_db:
            dbscan(params, vectors_np)
        if params.use_op:
            optics(params, vectors_np)

    if params.mode == "metadata":

        # write metadata
        if params.meta_labels:
            params.meta_labels_file = None

        write_meta(params)

    if params.mode == "tensorboard":

        # set up tensorboard projector
        tensorboard(params)
