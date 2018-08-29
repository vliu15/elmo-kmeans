# Goal
The goal of this pipeline is to take a dataset of voice transcriptions (obtained from patient surveys and logs) and perform topic analysis on it to find out what patients call in and talk about. These clusters of topics will be then used to help generate categories that will be used in Speech2Classify for classifying patient calls in real-time.

## Use Case(s)
 * Generate categories to classify patient calls

## Dataset(s)
 * Danita's 50k Medica dataset

## Approach
We combined deep learning and traditional NLP approaches to embed and cluster the transcriptions in the dataset. Below are details of each step that was implemented:

 1. Preprocessing the transcriptions:

  After compiling all the transcriptions into one file, we decided to embed per sentence, so we split each transcription by periods and put these sentences into a text file, each on its own line. Then we tokenized each sentence into a list of words for embedding.

  Approaches tried (to worse results):
    i. Transcription level (vs. sentence level)

  Approaches to try:
    i. Removing stop words (vs. leaving sentences as is)

 2. Word Embeddings:

  We take the lists of words and feed them into ELMo, a pretrained model that creates deep contextual representations of words. ELMo generates 3 vectors per word (each with varying levels of contextual dependency). To preserve contextual sentiment, we use strictly the third vector of each word.

  Approaches tried (to worse results):
    i.   GloVe (vs. ELMo)
    ii.  Averaging ELMo layers (vs. ELMo layer 3)

  Approaches to try:
    i. Trained ELMo (vs. Untrained ELMo)

  Links:
    i.  [ELMo Paper - AllenNLP](https://arxiv.org/pdf/1802.05365.pdf)
    ii. [ELMo GitHub - AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)
    iii. [GloVe - Stanford NLP](https://nlp.stanford.edu/projects/glove/)

 3. Sentence Embeddings:

  With a vector per word, we now have a matrix representation of each sentence. To create a sentence embedding from this matrix, we found that averaging all word vectors across the sentence yielded the most representative results. We then fed all embeddings through SIF, which removes calculates and removes the principal component of all the sentences.

  Approaches tried (to worse results):
    i.  Summing vs. concatenating (vs. averaging)
    ii. No SIF (vs. SIF)

  Approaches to try:
    i.  Max pooling (vs. averaging)

  Links:
    i.  [SIF Paper](https://openreview.net/pdf?id=SyK00v5xx)
    ii. [SIF GitHub](https://github.com/PrincetonML/SIF)

 4. Clustering:

  To find the optimal number of clusters for KMeans, we ran it iteratively over many different `k`'s (i.e. 10, 20, ..., 190, 200) and plotted the inertia, or the sum of the squared distance of all points to their respectivey clusters, against their corresponding `k`'s. The Elbow Method yielded no clear "elbow", or optimal `k`, so we tried plotting the silhouette scores against their corresponding `k`'s as well - that also did not yield clear results. We ended up taking the `k` value that was closest to being an "elbow" and ran KMeans to generate the clusters. We tried KMeans hierarchically as well (set `k`=2, then take each cluster of the output and repeat), which yielded a similar quality of results.

  Approaches tried (to worse results):
    i. MeanShift vs. DBSCAN (vs. KMeans, Hierarchical KMeans)

  Approaches to try:
    i. OPTICS vs. other clustering algorithms (vs. KMeans, Hierarchical KMeans)

  Links:
    i.   [Elbow Method - Wikipedia](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
    ii.  [Silhouette Scores - Wikipedia](https://en.wikipedia.org/wiki/Silhouette_(clustering))
    iii. [Silhouette Scores - SKLearn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
    iv.  [KMeans - Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
    v.   [KMEans - SKLearn](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

 5. Projecting to 3D:

  Because ELMo embeds in a 1024-dimensional space, we would like to project each embedding onto a 3-dimensional space for easier visualization. To do this, we ran t-SNE for about 5000 iterations (at perplexity=50) and converged relatively well.

  Approaches to try:
    i.  PCA (vs. t-SNE)
    ii. More/less iterations, higher/lower perplexity
