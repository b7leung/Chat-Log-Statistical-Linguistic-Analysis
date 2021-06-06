'''
Compute TFIDF encodings for training data
'''

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

parser = argparse.ArgumentParser(description='Compute feature vectors')
parser.add_argument('--datapath', type=Path, required=True,
                    help='Path to chats dataframe generated using preprocessing.py')
parser.add_argument('--minfreq', type=int, default=10,
                    help='Minimum appearences of a word to be included in dictionary')
parser.add_argument('--maxfeat', type=int, default=10000,
                    help='Maximum number of words to allow in dictionary')
parser.add_argument('--encoderfname', type=str, default='encoder.pkl',
                    help='File name to store encoder object')
parser.add_argument('--encodingsfname', type=str, default='encodings.pkl',
                    help='File name to store computed encodings of all data')
args = parser.parse_args()

if __name__ == '__main__':
    # df = pd.read_csv(args.datapath, converters={'Chats': eval})
    df = pickle.load(open(args.datapath, 'rb'))
    df['Chats'] = df['Chats'].apply(' '.join)
    vectorizer = TfidfVectorizer(min_df = args.minfreq, 
                                max_features = args.maxfeat,  
                                token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')
    tfidf_vectors = vectorizer.fit_transform(df['Chats'])

    with open(args.encoderfname, 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(args.encodingsfname, 'wb') as f:
        pickle.dump(tfidf_vectors, f)