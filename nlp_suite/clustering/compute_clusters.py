'''
Compute clusters using k-means
'''

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from pathlib import Path
import argparse
import pickle

parser = argparse.ArgumentParser(description='Compute clusters')
parser.add_argument('--encodings_path', type=Path, default='encodings.pkl',
                    help='Path to encodings pkl file generated using compute_tfidf.py')
parser.add_argument('--numclusters', type=int, default=6,
                    help='Number of clusters to compute')
parser.add_argument('--batchsize', type=int, default=16384,
                    help='Batch size to use for minibatch k-means algorithm')
parser.add_argument('--fname', type=str, default='cluster_data.pkl',
                    help='Name of output cluster data file')


def compute_clusters(encodings_path, num_clusters, batch_size):
    encodings = pickle.load(open(encodings_path, 'rb'))

    clusters = MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size)
    labels = clusters.fit_predict(encodings)
    pca3_proj = PCA(n_components=3).fit_transform(encodings.todense())

    return clusters, labels, pca3_proj


if __name__ == '__main__':
    args = parser.parse_args()
    clusters, labels, pca3_proj = compute_clusters(args.encodings_path, args.numclusters, args.batchsize)

    with open(args.fname, 'wb') as f:
        pickle.dump({'clusters': clusters,
                    'labels': labels,
                    'pca': pca3_proj},
                    f)
