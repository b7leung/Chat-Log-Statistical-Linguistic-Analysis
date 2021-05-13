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
parser.add_argument('--datapath', type=Path, required=True)
parser.add_argument('--numclusters', type=int, default=6)
parser.add_argument('--batchsize', type=int, default=16384)
parser.add_argument('--fname', type=str, default='cluster_data.pkl')
args = parser.parse_args()

if __name__ == '__main__':
    pkl_data = pickle.load(open(args.datapath, 'rb'))
    df, vectorizer, encodings = pkl_data['dataframe'], pkl_data['vectorizer'], pkl_data['encodings']

    clusters = MiniBatchKMeans(n_clusters=args.numclusters, batch_size=args.batchsize)
    labels = clusters.fit_predict(encodings)
    pca3_proj = PCA(n_components=3).fit_transform(encodings.todense())

    with open(args.fname, 'wb') as f:
        pickle.dump({'clusters': clusters,
                    'labels': labels,
                    'pca': pca3_proj},
                    f)
