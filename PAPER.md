# Summary
The goal of this pipeline is to take a dataset of voice transcriptions (obtained from patient surveys and logs) and perform topic analysis on it to find out what patients call in and talk about. These clusters of topics will be then used to help generate categories that will be used in Speech2Classify for classifying patient calls in real-time.

To do so, we used ELMo instead of traditional NLP approaches to create word and sentence embeddings, then used KMeans to generate clusters based on similarity in topics.

## Use Case(s)
 * Generate categories to classify patient calls

## Dataset(s)
 * Danita's 50k Medica dataset

## Approach
Below are details of each step that was implemented:

1. Preprocessing the transcriptions: After compiling all the transcriptions into one file, we decided to embed per sentence, so we split each transcription by periods and put these sentences into a text file, each on its own line. Then we tokenized each sentence into a list of words for embedding.  
    * Approaches tried (to worse results):
        - Transcription level (vs. sentence level)
    * Approaches to consider:
        - Removing stop words (vs. leaving sentences as is)

2. Word Embeddings: We take the lists of words and feed them into ELMo, a pretrained model that creates deep contextual representations of words. ELMo generates 3 vectors per word (each with varying levels of contextual dependency). To preserve contextual sentiment, we use strictly the third vector of each word.
    * Approaches tried (to worse results):
        - GloVe (vs. ELMo)
        - Averaging ELMo layers (vs. ELMo layer 3)
    * Approaches to consier:
        - Trained ELMo (vs. Untrained ELMo)
    * Links:
        - [ELMo Paper - AllenNLP](https://arxiv.org/pdf/1802.05365.pdf)
        - [ELMo GitHub - AllenNLP](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)
        - [GloVe - Stanford NLP](https://nlp.stanford.edu/projects/glove/)

3. Sentence Embeddings: With a vector per word, we now have a matrix representation of each sentence. To create a sentence embedding from this matrix, we found that averaging all word vectors across the sentence yielded the most representative results. We then fed all embeddings through SIF, which calculates and removes the principal component of all the sentences.
    * Approaches tried (to worse results):
        - Summing vs. concatenating (vs. averaging)
        - No SIF (vs. SIF)
    * Approaches to consider:
        - Max pooling (vs. averaging)
    * Links:
        - [SIF Paper](https://openreview.net/pdf?id=SyK00v5xx)
        - [SIF GitHub](https://github.com/PrincetonML/SIF)

4. Clustering: To find the optimal number of clusters for KMeans, we ran it iteratively over many different `k`'s (i.e. 10, 20, ..., 190, 200) and plotted the inertia, or the sum of the squared distance of all points to their respectivey clusters, against their corresponding `k`'s. The Elbow Method yielded no clear "elbow", or optimal `k`, so we tried plotting the silhouette scores against their corresponding `k`'s as well - that also did not yield clear results. We ended up taking the `k` value that was closest to being an "elbow" and ran KMeans to generate the clusters. We tried KMeans hierarchically as well (set `k`=2, then take each cluster of the output and repeat), which yielded a similar quality of results.
    * Approaches tried (to worse results):
        - MeanShift vs. DBSCAN (vs. KMeans, Hierarchical KMeans)
    * Approaches to consider:
        - OPTICS vs. other clustering algorithms (vs. KMeans, Hierarchical KMeans)
    * Links:
        - [Elbow Method - Wikipedia](https://en.wikipedia.org/wiki/Elbow_method_(clustering))
        - [Silhouette Scores - Wikipedia](https://en.wikipedia.org/wiki/Silhouette_(clustering))
        - [Silhouette Scores - SKLearn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
        - [KMeans - SKLearn](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

5. Projecting to 3D: Because ELMo embeds in a 1024-dimensional space, we would like to project each embedding onto a 3-dimensional space for easier visualization. To do this, we ran t-SNE for about 5000 iterations (at perplexity=50) and converged relatively well.
    * Approaches to try:
        - PCA (vs. t-SNE)
        - More/less iterations, higher/lower perplexity
    * Links:
        - [t-SNE - SKLearn](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
        - [t-SNE Tutorial](https://distill.pub/2016/misread-tsne/)

6. Visualizing: We use TensorBoard to render the embeddings. To see which sentences belong to which clusters, we created metadata which matches each embedding to its corresponding sentence and cluster. In TensorBoard, you are able to color the points based on their labels. It turns out that the projections generated from t-SNE are fall in loose clusters that are consistent with the output from our KMeans clustering. Our Kubernetes-powered TensorBoard is [here](dbslp1404:31313/#projector)
    * Directions to use:
        - In the upper left, adjust "Color by" to "Labels"
        - In the bottom left, there is the projectiong method: If you are rendering already-projected embeddings in 3D, then PCA will be sufficient. Otherwise, t-SNE does a better job at agglomerating clustered points together.

## Results
The pipeline delineated above has gotten us the best clusters so far. However, the clusters are far from perfect - it is often hard to find a common theme that defines a certain cluster; some clusters have outliers that do not seem to belong in their clusters.

The best standing results are [here](https://github.com/vliu15/elmo/tree/master/clusters) in the `clusters` folder in this repository. The pipeline to generate these clusters looks like:

```
   Preprocess at sentence level
-> Tokenize by period
-> Embed each with ELMo
-> Take the third vector of every word
-> Average all word vectors
-> Run through SIF, removing first principal component
-> Cluster with KMeans (`k`=100) and remove all clusters (specified in `remove.py`)
-> Cluster hierarchically with KMeans (split 6 times for 32 clusters)
```

## Related Work
 * Ayad Aliomer's [sentiment analysis](https://github.optum.com/AAT/sentiment-discovery)


## Contact

```
{
    Vincent Liu: vincent.liu@optum.com
    Dima Rekesh: dima.rekesh@optum.com
}
```
