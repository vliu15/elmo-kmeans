import numpy as np
import re
from allennlp.commands.elmo import ElmoEmbedder
from tqdm import *


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


def embed(params, sentences):
    '''
    Embed a list of sentences using ELMo or GloVe.
    :params.elmo: use ELMo
    :params.glove: use GloVe
    :params.concat_word_vecs: concatenate word vectors
    :params.sum_word_vecs: sum word vectors
    :params.avg_word_vecs: average word vectors
    :sentences: list of tokenized sentences/transcriptions to embed
    :return: a Tensor of Tensors (sentence/transcription embeddings)
    '''

    embeddings = []

    # embed with ELMo
    if params.elmo:
        elmo = ElmoEmbedder(params.elmo_options_file, params.elmo_weights_file, params.elmo_cuda_device)
        for i in tqdm(range(len(sentences))):
            emb = np.array(elmo.embed_sentence(sentences[i]), dtype=np.float32)
            embeddings.append(emb)

        # reduce word vectors: 3 -> 1
        reduce1 = []
        if params.bilm_layer_index == -1:
            for sentence in tqdm(embeddings):
                reduce1.append(np.mean(sentence, axis=0))
        elif params.bilm_layer_index <= 2 and params.bilm_layer_index >= 0:
            for sentence in tqdm(embeddings):
                bilm_layer = sentence[params.bilm_layer_index, :, :]
                bilm_layer = np.squeeze(bilm_layer, axis=0)
                reduce1.append(bilm_layer) # shape: [n, 1024]

    # embed with GloVe
    elif params.glove:
        word_dict = load_glove(params.glove_word_file)
        reduce1 = []
        for i in tqdm(range(len(sentences))):
            embeddings = np.stack([word_dict[word]
                                   if word in word_dict.keys()
                                   else word_dict['OOV']
                                   for word in sentences[i]
                                  ], axis=0)
            reduce1.append(embeddings)

    # reduce sentence vectors -> 1
    reduce2 = []
    if params.avg_word_vecs:
        for sentence in tqdm(reduce1):
            reduce2.append(np.mean(sentence, axis=0))
    elif params.max_pool_word_vecs:
        for sentence in tqdm(reduce1):
            reduce2.append(np.amax(sentence, axis=0))
    elif params.concat_word_vecs:
        for sentence in tqdm(reduce1):
            reduce2.append(concat_word_vecs(sentence, params.max_transcript_len))
    elif params.sum_word_vecs:
        for sentence in tqdm(reduce1):
            reduce2.append(np.sum(sentence, axis=0))

    # convert to a tensor of tensors
    return np.stack([x for x in reduce2], axis=0)


def tokenize(params):
    '''
    Tokenize a file per line by space ' '.
    :params.sentence_file: file to be tokenized
    :return: list of lists of tokens (per sentence/transcription)
    '''

    # with open(params.sentence_file, 'r', encoding=params.encoding, errors=params.errors) as f:
    with open(params.sentence_file, 'r') as f:

        # convert each sentence into list of tokens
        tokenized = []
        for s in f:
            s = s.replace("'", "")
            tokenized.append(re.findall(r"[\w]+|[.,!?;:()%$&]", s))
            # tokenized.append(s.split(' '))

    return tokenized
