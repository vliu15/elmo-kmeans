import tensorflow as tf
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from tqdm import *

# embeds a list of tokenized sentences
# returns a numpy array of numpy sentence vectors
def embed(ElmoEmbedder, sentences):

    # embed each sentence with ELMo
    embeddings = []
    for i in tqdm(range(len(sentences))):
        embeddings.append(ElmoEmbedder.embed_sentence(sentences[i])) # sentence shape: [3, n, 1024]

    # reduce number of vectors per word: 3 -> 1
    reduce1 = []
    for sentence in embeddings:
        # bilm_layer3 = tf.slice(sentence, [2, 0, 0], [1, -1, -1])
        # bilm_layer3 = tf.squeeze(bilm_layer3, axis=0)
        # reduce1.append(bilm_layer3) # sentence shape: [n, 1024]
        reduce1.append(tf.reduce_mean(sentence, axis=0)) # sentence shape: [n, 1024]

    # reduce sentence matrix, per sentence
    reduce2 = []
    for sentence in reduce1:
        reduce2.append(tf.reduce_sum(sentence, axis=0)) # sentence shape: [1024]

    # convert to a tensor of tensors
    reduce2 = tf.stack([x for x in reduce2])

    return reduce2


# cleans and tokenizes transcription file, already separated by line
def tokenize(transcript_file, sentences_file):

    with open(transcript_file, 'r') as f:

        # read file line by line
        text = f.read().splitlines()

        # remove first line of labels
        del text[0]

        # create list of all sentences
        sentences = []
        for file in text:
            transcription = file.split(',')[2:]
            transcription = ','.join(transcription)
            sentences.extend(transcription.split('.'))

        # filter out empty entries
        sentences = list(filter(None, sentences))

    with open(sentences_file, 'w') as f:

        # convert each sentence into list of tokens
        tokenized = []
        for s in sentences:
            x = s.lstrip(' ') # remove leading ws
            f.write(x + '\n') # save a copy of all sentences
            x = x.split(' ') # tokenize
            x = x + ['.'] # hard-code append '.'
            tokenized.append(x)

    return tokenized
