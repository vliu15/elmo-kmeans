import tensorflow as tf
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from tqdm import *

# concatenates word vectors together
def concat_word_vecs(sentence_vec, max_len):

    # pad/truncate to same shape
    sentence_len = sentence_vec.get_shape().as_list()[0]
    if sentence_len > max_len:
        sentence_vec = tf.slice(sentence_vec, [0, 0], [max_len, -1])
    elif sentence_len < max_len:
        padding = tf.constant([[0, max_len - sentence_len], [0, 0]])
        sentence_vec = tf.pad(sentence_vec, padding)
    
    return tf.concat([sentence_vec[i] for i in range(max_len)], axis=0)

# returns a NumPy array of sentence vectors
def embed(params, ElmoEmbedder, sentences):

    # embed each sentence with ELMo
    embeddings = []
    for i in tqdm(range(len(sentences))):
        embeddings.append(ElmoEmbedder.embed_sentence(sentences[i])) # shape: [3, n, 1024]


    # reduce word vectors: 3 -> 1
    reduce1 = []

    # average
    if params.bilm_layer_index == -1:
        for sentence in embeddings:
            reduce1.append(tf.reduce_mean(sentence, axis=0))

    # bilm layer
    elif params.bilm_layer_index <= 2 and params.bilm_layer_index >= 0:
        for sentence in embeddings:
            bilm_layer = tf.slice(sentence, [params.bilm_layer_index, 0, 0], [1, -1, -1])
            bilm_layer = tf.squeeze(bilm_layer, axis=0)
            reduce1.append(bilm_layer) # shape: [n, 1024]


    # reduce sentence matrix, per sentence
    reduce2 = []

    # concatenate word vectors
    if params.concat_word_vecs:
        for sentence in reduce1:
            reduce2.append(concat_word_vecs(sentence, params.max_transcript_len))

    # sum word vectors
    elif params.sum_word_vecs:
        for sentence in reduce1:
            reduce2.append(tf.reduce_sum(sentence, axis=0))



    # convert to a tensor of tensors
    return tf.stack([x for x in reduce2])


# cleans and tokenizes transcription file, already separated by line
def tokenize(params):

    with open(params.sentence_file, 'r') as f:

        # read file line by line
        text = f.read().splitlines()

        # convert each sentence into list of tokens
        tokenized = []
        for s in text:
            tokenized.append(s.split(' '))

    return tokenized
