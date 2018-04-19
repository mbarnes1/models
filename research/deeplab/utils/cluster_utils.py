"""
Rounding schemes.
"""

import numpy as np
import random


def kwik_cluster(V, cost_function, blocks=None, normalize=True):
    """
    KwikCluster (Ailon2008) based on cosine similarity
    :param V:               N x D numpy array where N is number of pixels and D is the embedding dimension.
    :param cost_function:   Function that maps vector dot product (cosine similarity) to a cost for creating graph.
    :param blocks:          N numpy array, specifying the block labels. Only samples within the same block can be
                            clustered together.
    :param normalize:       Normalize pixel embeddings to have norm 1.
    :return labels:         N numpy vector with predicted labels for each pixel.
    """
    n, d = V.shape
    labels = -1*np.ones(n)
    unlabeled_indices = np.array(xrange(0, n))
    counter = 0

    if normalize:
        l2_norms = np.linalg.norm(V, axis=1)
        assert np.all(np.greater(l2_norms, 0.))
        V = np.divide(V, l2_norms[:, None])
        assert np.all(np.isfinite(V))

    while unlabeled_indices.size > 0:  # until all samples are labeled
        pivot_index = random.sample(unlabeled_indices, 1)[0]
        pivot = V[pivot_index]
        cosine_similarities = np.dot(V, pivot)  # N entries
        probability = cost_function(cosine_similarities)
        cluster = np.greater(probability, np.random.uniform(size=probability.shape))
        cluster[pivot_index] = True

        # If already in a cluster, do not reassign
        labeled_indices_mask = np.not_equal(labels, -1)
        cluster[labeled_indices_mask] = 0

        # If in different block, do not assign
        if blocks is not None:
            pivot_block = blocks[pivot_index]
            cluster[blocks != pivot_block] = 0

        labels[cluster] = counter
        counter += 1

        unlabeled_indices = np.nonzero(np.equal(labels, -1))[0]
    return labels


def lp_cost(x, p=2):
    """
    :param x: Numpy vector of dot products (i.e. 1 - cosine similarity)
    :param p: Power to raise x
    :return:  Cost of placing these two samples in different clusters. In [0, 1]
    """
    x = np.maximum(x, 0.)
    num = np.power(x, p)
    den = num + np.power((1.0 - x), p)
    return np.divide(num, den)
