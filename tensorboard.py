import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import os

LOG_DIR = os.path.join(os.getcwd(), "output")

def tensorboard(embeddings_file, metadata_file):

    tmp = np.load(embeddings_file)
    embedding_var = tf.Variable(tmp, trainable=False, name="embeddings")

    config = projector.ProjectorConfig()

    # add one embedding
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # link to metadata
    embedding.metadata_path = metadata_file

    # write to LOG_DIR
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # creates a projector_config.pbtxt in LOG_DIR
    projector.visualize_embeddings(summary_writer, config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "embed.ckpt"), 1)
