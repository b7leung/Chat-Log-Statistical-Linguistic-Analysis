import os
from pathlib import Path
import argparse
import pickle
from text_analysis_kevin import get_cluster_analysis
import itertools as it
import pandas as pd
import errno

parser = argparse.ArgumentParser(description='Preprocess discord chat data')
parser.add_argument('--user_dataframe_path', type=Path, required=True,
        help='Path to user chat dataframe pkl file generated by cluster preprocessing')
parser.add_argument('--cluster_path', type=Path, required=True,
        help='Path to clustering data pkl file')
parser.add_argument('--clusters', nargs='+', type=int, default=[],
        help='Specify which clusters to analyse; computes all if not specified')
parser.add_argument('--word_threshold', type=int, default=10000000,
        help='Cluster will be randomly sampled if the word cout exceeds this value')
parser.add_argument('--verbose', action='store_true', 
        help='Print script progress')
args = parser.parse_args()

if __name__ == '__main__':
    user_df = pickle.load(open(args.user_dataframe_path, 'rb'))
    labels = pickle.load(open(args.cluster_path, 'rb'))['labels']

    if args.verbose:
        print('User data and cluster labels loaded succesfully.')

    keys = 'message_lengths,average_word_lengths,stop_dic,unigrams,bigrams,trigrams,wordcloud'.split(',')

    clusters_to_compute = set(labels) if not args.clusters else args.clusters

    for i in clusters_to_compute:
        if args.verbose:
            print(f'Computing analysis data for cluster {i}')
        cluster_df = user_df.iloc[labels==i]
        cluster_words = cluster_df['Chats'].apply(' '.join).str.count(' ').sum()
        if cluster_words/args.word_threshold > 1:
            cluster_df = cluster_df.sample(frac=args.word_threshold/cluster_words, random_state=1)
            if args.verbose:
                print(f'Cluster contained {cluster_words} words. Cluster downsampled.')

        text_analysis = get_cluster_analysis(cluster_df)
        analysis_dict = dict(zip(keys, text_analysis))
        with open(f'cluster{i}_analysis.pkl', 'wb') as f:
            pickle.dump(analysis_dict, f)

        if args.verbose:
            print(f'Successfully saved cluster {i} data in cluster{i}_analysis.pkl')

