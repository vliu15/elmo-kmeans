#!bin/bash

MODEL_DIR="$PWD/model"
DATA_DIR="$PWD/data"

mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR

# wget to install options and weights for ELMo
wget -o $MODEL_DIR/options.json https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json

wget -o $MODEL_DIR/weights.hdf5 https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5

# wget to install GloVe files
wget -o $MODEL_DIR/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip && unzip $MODEL_DIR/glove.840B.300d.zip -d $MODEL_DIR

# RUN wget -o $MODEL_DIR/glove.840B.300d-char.txt https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt

# wget to install benchmark files
wget -o $DATA_DIR http://statmt.org/wmt11/training-monolingual-news-2011.tgz && \
    gunzip $DATA_DIR/training-monolingual-news-2011.tgz && \
    tar -xvf training-monolingual-news-2011.tar && \
    

