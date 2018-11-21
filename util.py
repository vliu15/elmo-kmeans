import numpy as np
import re
from allennlp.commands.elmo import ElmoEmbedder
from tqdm import *


def file_len(f):
    """
    Returns number of lines in a file.
    :f: file object reader
    """

    for n, l in enumerate(f, 1):
        pass
    f.seek(0) # rewind
    return n

def concat_word_vecs(sentence_vec, max_len=50):
    '''
    Concatenate word embeddings together to get sentence/transcription vector.
    :sentence_vec: 2D Tensor of word embeddings
    :max_len: length of each sentence to pad/truncate to
    :return: a 1D Tensor for the sentence/transcription
    '''

    # pad/truncate to same shape
    sentence_len = sentence_vec.shape[0]
    if sentence_len > max_len:
        sentence_vec = sentence_vec[0:max_len, :]
    elif sentence_len < max_len:
        pad_width = np.array([[0, max_len - sentence_len], [0, 0]])
        pad_values = np.array([[0, 0], [0, 0]])
        sentence_vec = np.pad(sentence_vec, pad_width, 'constant', constant_values=pad_values)
    
    return np.concat([sentence_vec[i] for i in range(max_len)], axis=0)


def load_glove(glove_file):
    '''
    Create a dictionary for GloVe lookup.
    :glove_file: location of GloVe text file
    :return: a Python dictionary of GloVe word embeddings
    '''
    with open(glove_file, 'r', encoding="utf-8") as f:
        word_dict = {}
        lines = f.read().splitlines()
        for i in tqdm(range(len(lines))):
            line = lines[i].split()
            word = ''.join(line[:-300])
            embedding = np.asarray(line[-300:], dtype='float32')
            word_dict[word] = embedding
    return word_dict


def embed(params):
    '''
    Embed a list of sentences using ELMo or GloVe.
    :params.elmo: use ELMo
    :params.glove: use GloVe
    :params.concat_word_vecs: concatenate word vectors
    :params.sum_word_vecs: sum word vectors
    :params.avg_word_vecs: average word vectors
    :return: a Tensor of Tensors (sentence/transcription embeddings)
    '''

    def compress(emb):
        """
        Compress a matrix of word vectors into a sentence vector.
        """
        if params.sum_word_vecs:
            return np.sum(emb, axis=0)
        if params.max_pool_word_vecs:
            return np.amax(emb, axis=0)
        if params.concat_word_vecs:
            return concat_word_vecs(emb, params.max_transcript_len)
        if params.avg_word_vecs:
            return np.mean(emb, axis=0)

    f = open(params.sentence_file, 'r', encoding=params.encoding, errors=params.errors)
    # f = open(params.sentence_file, 'r')
    num = file_len(f)

    # initialize embedding methods
    if params.elmo:
        elmo = ElmoEmbedder(params.elmo_options_file, params.elmo_weights_file, params.elmo_cuda_device)
    if params.glove:
        word_dict = load_glove(params.glove_word_file)

    e_emb, g_emb = [], []

    # tokenize each line
    for i, s in tqdm(enumerate(f, 1), total=num):
        s = s.replace("'", "")
        s = re.findall(r"[\w]+|[.,!?;:()%$&#]", s)

        # embed with ELMo
        if params.elmo:

            # write to prevent OOM
            if i % 500000 == 0:
                np.save('elmo%d_'%(i//500000) + params.embedding_file, np.stack(e_emb, axis=0))
                print('Wrote to elmo%d_'%(i//500000) + params.embedding_file)
                e_emb = []

            emb = np.array(elmo.embed_sentence(s), dtype=np.float32)

            # reduce word vectors: 3 -> 1
            if params.bilm_layer_index == -1:
                emb = np.mean(emb, axis=0)
            elif params.bilm_layer_index <= 2 and params.bilm_layer_index >= 0:
                emb = emb[params.bilm_layer_index, :, :] # shape: [n, 1024]

            # reduce sentence vectors -> 1
            e_emb.append(compress(emb))

        # embed with GloVe
        if params.glove:

            # write to prevent OOM
            if i % 500000 == 0:
                np.save('glove%d_'%(i//500000) + params.embedding_file, np.stack(e_emb, axis=0))
                print('Wrote to glove%d_'%(i//500000) + params.embedding_file)
                g_emb = []

            emb = np.stack([word_dict[word]
                                   if word in word_dict.keys()
                                   else word_dict['OOV']
                                   for word in s
                                  ], axis=0)

            # reduce sentence vectors -> 1
            g_emb.append(compress(emb))

    if len(e_emb) > 0:
        np.save('elmo%d_'%(i//500000) + params.embedding_file, np.stack(e_emb, axis=0))
        print('Wrote to elmo%d_'%(i//500000) + params.embedding_file)
    if len(g_emb) > 0:
        np.save('glove%d_'%(i//500000) + params.embedding_file, np.stack(g_emb, axis=0))
        print('Wrote to glove%d_'%(i//500000) + params.embedding_file)
