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
parser.add_argument('--encoder_path', type=Path, default='encoder.pkl',
                    help='Path to encoder pkl file generated using compute_tfidf.py')
parser.add_argument('--encodings_path', type=Path, default='encodings.pkl',
                    help='Path to encodings pkl file generated using compute_tfidf.py')
parser.add_argument('--numclusters', type=int, default=6,
                    help='Number of clusters to compute')
parser.add_argument('--batchsize', type=int, default=16384,
                    help='Batch size to use for minibatch k-means algorithm')
parser.add_argument('--fname', type=str, default='cluster_data.pkl',
                    help='Name of output cluster data file')
args = parser.parse_args()

if __name__ == '__main__':
    vectorizer = pickle.load(open(args.encoder_path, 'rb'))
    encodings = pickle.load(open(args.encodings_path, 'rb'))

    clusters = MiniBatchKMeans(n_clusters=args.numclusters, batch_size=args.batchsize)
    labels = clusters.fit_predict(encodings)
    pca3_proj = PCA(n_components=3).fit_transform(encodings.todense())

    with open(args.fname, 'wb') as f:
        pickle.dump({'clusters': clusters,
                    'labels': labels,
                    'pca': pca3_proj},
                    f)
