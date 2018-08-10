FROM docker.optum.com/dl_lab/elmo:vincent_v1

RUN apt-get install wget && apt-get install update

RUN mkdir /home
WORKDIR /home
ADD . /home
RUN mkdir /home/model

# wget to install options and weights for ELMo
RUN wget -o /home/model/options.json https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json

RUN wget -o /home/model/weights.hdf5 https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
