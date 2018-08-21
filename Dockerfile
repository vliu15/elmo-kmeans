FROM docker.optum.com/dl_lab/elmo:tf-gpu

RUN apt-get install wget && apt-get install update

RUN mkdir /home
WORKDIR /home
ADD . /home
RUN mkdir /home/model
MODEL_DIR=/home/model

# wget to install options and weights for ELMo
RUN wget -o $MODEL_DIR/options.json https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json

RUN wget -o $MODEL_DIR/weights.hdf5 https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5

# wget to install GloVe files
RUN wget -o $MODEL_DIR/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip && unzip $MODEL_DIR/glove.840B.300d.zip -d $MODEL_DIR

RUN wget -o $MODEL_DIR/glove.840B.300d-char.txt https://raw.githubusercontent.com/minimaxir/char-embeddings/master/glove.840B.300d-char.txt
