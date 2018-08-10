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
weight_file = os.path.join(os.getcwd(), "model", "weights.hdf5")

# command line argument: file name (all.csv)
parser = argparse.ArgumentParser(description="arguments for ELMo embeddings")
parser.add_argument('-e', '--embed', type=str, help='embed sentences')
parser.add_argument('-c', '--cluster', type=str, help='use clustering algorithm')
parser.add_argument('-v', '--tsne', help='use tsne to 3-d', action='store_true')
parser.add_argument('-t', '--tensorboard', help='use tensorboard for visualization', action='store_true')
args = parser.parse_args()

# output files
output_dir = os.path.join(os.getcwd(), "output", "test")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
transcription_file = args.embed
sentence_file = os.path.join(output_dir, "sentences.txt")
embedding_file = os.path.join(output_dir, "embeddings.npy")
ms_labels_file = os.path.join(output_dir, "ms_labels.json")
db_labels_file = os.path.join(output_dir, "db_labels.json")
op_labels_file = os.path.join(output_dir, "op_labels.json")
ts_embed_file = os.path.join(output_dir, "embeddings_ts.npy")
metadata_file = os.path.join(output_dir, "metadata.tsv")


if __name__ == "__main__":

    if args.embed:
        # tokenize transcriptions
        tokenized = tokenize(transcription_file, sentence_file)

        # set up embedding with ELMo
        elmo = ElmoEmbedder(options_file, weight_file)

        # set up TF with GPU usage
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            emb = sess.run(embed(elmo, tokenized))
            np.save(embedding_file, emb)

    if args.cluster:
        # load embeddings.npy file
        vectors_np = np.load(embedding_file)

        # 3-d visualization
        if args.tsne:
            vectors_np = tsne(vectors_np, ts_embed_file)

        # perform clustering
        labels_file = None
        if args.cluster == 'meanshift':
            labels_file = ms_labels_file
            mean_shift(vectors_np, labels_file)
        elif args.cluster == 'dbscan':
            labels_file = db_labels_file
            dbscan(vectors_np, labels_file)
        elif args.cluster == 'optics':
            labels_file = op_labels_file
            optics(vectors_np, labels_file)

        # write metadata
        if not labels_file == None:
            write_meta(sentence_file, labels_file, metadata_file)

    if args.tensorboard:
        # set up tensorboard projector
        tensorboard(embeddings_file, metadata_file)
