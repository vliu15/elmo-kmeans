import numpy as np
from sklearn.manifold import TSNE

def tsne(params):

    vectors = np.load(params.embedding_file)

    ts = TSNE(n_components=params.ts_n_components,
              perplexity=params.ts_perplexity,
              learning_rate=params.ts_learning_rate,
              n_iter=params.ts_n_iter)

    vectors_ts = ts.fit_transform(vectors)

    np.save(params.tsne_file, vectors_ts)
