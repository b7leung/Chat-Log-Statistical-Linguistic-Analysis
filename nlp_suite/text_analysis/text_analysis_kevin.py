import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict, Counter
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import itertools as it

def get_text_analysis(df):
    '''
    Get Stop Words histogram, Unigrams, Bigrams, Trigra, and Word Cloud
    Input: 
    df: Datatframe with a column labeled 'user_messages' 
    
    Outputs: 
    message_lengths: pd.Series of each message's length in terms of words
    average_word_length: pd.Series of each message's average word length
    stop_dic : collections.defaultdict
    unigrams: decending ordered by frequency list of tuples: (unigram, frequency)
    bigrams: decending ordered by frequency list of tuples: (bigram, frequency)
    trigrams: decending ordered by frequency list of tuples: (trigram, frequency)
    wordcloud: word cloud function, plot using plt.imshow
    '''
    nltk.download('stopwords')
    stop=set(stopwords.words('english'))
    corpus=[]
    new= df['user_messages'].str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    stop_dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            stop_dic[word]+=1

    counter=Counter(corpus)
    most=counter.most_common()
    x, y= [], []
    for word,count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)
    corpus = pd.Series(preprocess_news(df))
    corpus = corpus.apply(lambda x:' '.join(x))
    unigrams=get_top_ngram(corpus,n=1)
    bigrams=get_top_ngram(corpus,n=2)
    trigrams=get_top_ngram(corpus,n=3)
    wordcloud = get_wordcloud(corpus)
    message_lengths = df['user_messages'].str.len()
    average_word_lengths = df['user_messages'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))
    return message_lengths,average_word_lengths,stop_dic,unigrams,bigrams,trigrams,wordcloud



WORD = re.compile(r'\w+')
def regTokenize(text):
    words = WORD.findall(text)
    return words

def get_cluster_analysis(df):
    '''
    
    '''
    nltk.download('stopwords', quiet=True)
    stop=set(stopwords.words('english'))

    corpus=[]
    chats = pd.Series(it.chain(*df['Chats']))
    chats.dropna(inplace = True)
    
    corpus=chats.apply(regTokenize)
    corpus = corpus.apply(lambda x:' '.join(x))
    stop_dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            stop_dic[word]+=1
    counter=Counter(corpus)
    most=counter.most_common()
    x, y= [], []
    for word,count in most[:40]:
        if (word not in stop):
            x.append(word)
            y.append(count)
    
    unigrams=get_top_ngram(chats,n=1)
    bigrams=get_top_ngram(chats,n=2)
    trigrams=get_top_ngram(chats,n=3)
    wordcloud = get_wordcloud(chats)
    message_lengths = chats.str.len()
    average_word_lengths = chats.str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x))
    return message_lengths,average_word_lengths,stop_dic,unigrams,bigrams,trigrams,wordcloud



def get_top_ngram(corpus, n=None):
    '''
    Find ngram frequency in descending order
    '''
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq

def get_wordcloud(data):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=100,
        max_font_size=30,
        scale=3,
        random_state=1)
   
    wordcloud=wordcloud.generate(str(data))
    return wordcloud

def plot_message_lengths_hist(message_lengths):
    plt.hist(message_lengths,bins = int(max(message_lengths)//10),density=True)
    plt.xlim([0,1000])
    plt.title('Message Lengths Histogram')
    plt.savefig('plots/message_lengths_hist.png',bbox_inches='tight',dpi=1000)

def plot_average_word_lengths(average_word_lengths):
    plt.hist(average_word_lengths,bins = int(max(average_word_lengths)))
    plt.xlim([0,50])
    plt.title('Average Word Length Histogram')
    plt.savefig('plots/average_word_lengths_hist.png',bbox_inches='tight',dpi=1000)

def plot_stop_dic(stop_dic,top_n=10):
    idx = np.argsort(list(stop_dic.values()))[::-1]
    words = np.array(list(stop_dic.keys()))
    freq = np.array(list(stop_dic.values()))
    plt.bar(words[idx[:top_n]], freq[idx[:top_n]])
    plt.title('Top 10 Stop Words Histogram')
    plt.savefig('plots/stop_dic_histogram.png',bbox_inches='tight',dpi=1000)

def plot_top_n_ngrams(ngrams,n=10):
    x,y = zip(*ngrams)
    plt.barh(list(x[:n]),list(y[:n]))

def preprocess_news(df):
    stop=set(stopwords.words('english'))
    nltk.download('punkt')
    nltk.download('wordnet')
    corpus=[]
    stem=PorterStemmer()
    lem=WordNetLemmatizer()
    for news in df['user_messages']:
        words=[w for w in word_tokenize(news) if (w not in stop)]
        
        words=[lem.lemmatize(w) for w in words if len(w)>2]
        
        corpus.append(words)
    return corpus

def plot_word_cloud(wordcloud):
    plt.imshow(wordcloud)
    plt.title('Word Cloud')
    plt.axis('off')
    plt.savefig('plots/word_cloud.png',bbox_inches='tight',dpi=1000)

def plot_unigrams(unigrams):
    plot_top_n_ngrams(unigrams)
    plt.title('Top 10 Unigrams Histogram')
    plt.savefig('plots/unigrams.png',bbox_inches='tight',dpi=1000)

def plot_bigrams(bigrams):
    plot_top_n_ngrams(bigrams)
    plt.title('Top 10 Bigrams Histogram')
    plt.savefig('plots/bigrams.png',bbox_inches='tight',dpi=1000)

def plot_trigrams(trigrams):
    plot_top_n_ngrams(trigrams)
    plt.title('Top 10 Trigrams Histogram')
    plt.savefig('plots/trigrams.png',bbox_inches='tight',dpi=1000)

def get_plots(message_lengths,average_word_lengths,stop_dic,unigrams,bigrams,trigrams,wordcloud):
    if not os.path.exists('plots'):
        os.mkdir('plots')
    plt.style.use('dark_background')
    plt.figure()
    plot_word_cloud(wordcloud)
    plt.figure()
    plot_message_lengths_hist(message_lengths)
    plt.figure()
    plot_unigrams(unigrams)
    plt.figure()
    plot_average_word_lengths(average_word_lengths)
    plt.figure()
    plot_bigrams(bigrams)
    plt.figure()
    plot_stop_dic(stop_dic)
    plt.figure()
    plot_trigrams(trigrams)
    plt.figure()