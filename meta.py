import json

def write_meta(params):

    if not params.meta_labels_file == None:
        sentences = [sentence.rstrip('\n') for sentence in open(params.sentence_file)]

        with open(params.meta_labels_file, 'r') as f:
            labels = json.load(f) # from labels generated by clustering

        meta = ['Sentence\tLabel\n']
        meta.extend([s + '\t' + str(l) + '\n' for s, l in zip(sentences, labels)])
    
        with open(params.meta_file, 'w') as f:
            f.writelines(meta)

    elif params.meta_labels_file == None:
        with open(params.sentence_file, 'r') as f:
            contents = f.read()
        with open(params.meta_file, 'w') as f:
            f.write(contents)
