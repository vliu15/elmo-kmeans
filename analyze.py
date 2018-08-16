from tqdm import *

import shutil
import json
import os

def write_groups(params):

    with open(params.sentence_file, 'r') as f:
        sentences = f.read().splitlines()

    with open(params.group_labels_file, 'r') as f:
        labels = json.load(f)

    shutil.rmtree(params.group_dir)
    os.makedirs(params.group_dir)

    for i in tqdm(range(len(labels))):
        label = labels[i]
        file_name = str(label) + '.txt'
        with open(os.path.join(params.group_dir, file_name), 'a') as f:
            f.write(sentences[i] + '\n')
