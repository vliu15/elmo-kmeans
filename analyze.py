from tqdm import *

import numpy as np
import shutil
import json
import os

def write_groups(params):
    '''
    Write sentences to .txt files correpsonding to cluster labels.
    :params.sentence_file: file of all sentences
    :params.group_labels_file: file of labels of all sentences
    '''

    if not os.exists(params.group_dir):
        os.makedirs(params.group_dir)

    with open(params.sentence_file, 'r') as f:
        sentences = f.read().splitlines()

    with open(params.group_labels_file, 'r') as f:
        labels = json.load(f)

    # clear current directory of groups
    shutil.rmtree(params.group_dir)
    os.makedirs(params.group_dir)

    # write labels to .txt files
    for i in tqdm(range(len(labels))):
        label = labels[i]
        file_name = str(label) + '.txt'
        with open(os.path.join(params.group_dir, file_name), 'a') as f:
            f.write(sentences[i] + '\n')

def remove_groups(params):
    '''
    Remove sentences from specified groups.
    :params.embedding_file: file of embeddings to be adjusted
    :params.sentences_file: file of sentences to be adjusted
    '''

    if not os.exists(params.trim_dir):
        os.makedirs(params.trim_dir)

    trim_emb_file = os.path.join(params.trim_dir, "embeddings.npy")
    trim_sent_file = os.path.join(params.trim_dir, "sentences.txt")

    # labels to be removed
    remove = []

    embeddings = np.load(params.embedding_file)

    with open(params.sentences_file, 'r') as f:
        sentences = f.read().splitlines()

    with open(params.groups_labels_file, 'r') as f:
        labels = json.load(f)

    # create trimmed copies of embeddings and sentences
    emb, sent = [embeddings[i], sentences[i] for i in labels
                            if not i in remove]

    np.save(trim_emb_file, np.stack(emb))
    with open(trim_sent_file, 'w') as f:
        f.writelines(sent)
