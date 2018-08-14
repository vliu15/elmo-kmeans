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

# returns dictionary of GloVe word embeddings
def load_glove(glove_file):
    with open(glove_file, 'r', encoding="utf-8") as f:
        word_dict = {}
        lines = f.read().splitlines()
        for i in tqdm(range(len(lines))):
            line = lines[i].split()
            word = ''.join(line[:-300])
            embedding = np.asarray(line[-300:], dtype='float32')
            word_dict[word] = embedding
    return word_dict

# returns a NumPy array of sentence vectors
def embed(params, sentences):

    # embed transcriptions
    embeddings = []

    # embed with ELMo
    if params.elmo:
        elmo = ElmoEmbedder(params.elmo_options_file, params.elmo_weights_file)
        for i in tqdm(range(len(sentences))):
            embeddings.append(elmo.embed_sentence(sentences[i])) # shape: [3, n, 1024]

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

    # embed with GloVe
    elif params.glove:
        word_dict = load_glove(params.glove_word_file)
        reduce1 = []
        oov = 0
        for i in tqdm(range(len(sentences))):
            embeddings = tf.stack([word_dict[word]
                                   if word in word_dict.keys()
                                   else word_dict['OOV']
                                   for word in sentences[i]
                                  ])
            for i in range(embeddings.get_shape().as_list()[0]):
                if embeddings[i] == word_dict['OOV']:
                    oov += 1
            reduce1.append(embeddings)
        print("Number of OOV words: %d\n" % oov)


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
    all = tf.stack([x for x in reduce2])
    print(all.get_shape().as_list())
    return all


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
