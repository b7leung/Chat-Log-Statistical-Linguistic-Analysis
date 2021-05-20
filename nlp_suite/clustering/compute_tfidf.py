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
parser.add_argument('--datapath', type=Path, required=True)
parser.add_argument('--minfreq', type=int, default=10)
parser.add_argument('--maxfeat', type=int, default=10000)
parser.add_argument('--fname', type=str, default='encoded_data.pkl')
args = parser.parse_args()

if __name__ == '__main__':
    # df = pd.read_csv(args.datapath, converters={'Chats': eval})
    df = pickle.load(open(args.datapath, 'rb'))
    df['Chats'] = df['Chats'].apply(' '.join)
    vectorizer = TfidfVectorizer(min_df = args.minfreq, 
                                max_features = args.maxfeat,  
                                token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')
    tfidf_vectors = vectorizer.fit_transform(df['Chats'])

    with open(args.fname, 'wb') as f:
        pickle.dump({
                    # 'dataframe': df,
                    'vectorizer': vectorizer,
                    'encodings': tfidf_vectors},
                    f)
