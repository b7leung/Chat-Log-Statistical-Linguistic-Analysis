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
    '''Get text analysis for a spcific user's messages.
    Input should include a 'user_messages' column which has 
    one message of the user per row
    
    :param df: Datatframe with a column labeled 'user_messages' 
    :type df: pandas.DataFrame
    :return: tuple of analysis 
    :rtype: tuple
    '''
    #(message_lengths,average_word_lengths,stop_dic,unigrams,bigrams,trigrams,wordcloud)
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
    ''' Tokenizes text

    :param text: text to be tokenized
    :type text: string
    '''
    words = WORD.findall(text)
    return words

def get_cluster_analysis(df):
    '''Get text analysis on an entire cluster. 
    This is similar to get text analysis but for multiple 
    users instead of just one user

    :param df: dataframe with the user's messages
    :type df: pandas.DataFrame
    :return: tuple of analysis
    :rtype: tuple
    '''    
    #  (message_lengths,average_word_lengths,stop_dic,unigrams,bigrams,trigrams,wordcloud)
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
    '''Find ngram frequency in descending order

    :param corpus: tokenized words
    :type corpus: list
    :param n: number of words to get frequencies (n in ngram), defaults to None
    :type n: int, optional
    :return: the frequncy of the corresopnding ngram (ngram, frequency) in decencing order by frequency
    :rtype: list of tuple
    '''    
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq

def get_wordcloud(data):
    '''Creates a wordcloud from the given data 

    :param data: words to be used for creating wordcloud
    :type data: pd.Series
    :return: wordcloud
    :rtype: wordcloud.WordCloud
    '''    
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
    '''plots and saves the message lengths histogram

    :param message_lengths: lengths of messages
    :type message_lengths: pd.Series
    '''    
    plt.hist(message_lengths,bins = int(max(message_lengths)//10),density=True)
    plt.xlim([0,1000])
    plt.title('Message Lengths Histogram')
    plt.savefig('plots/message_lengths_hist.png',bbox_inches='tight',dpi=1000)

def plot_average_word_lengths(average_word_lengths):
    '''plots and saves the average word lengths histogram per message

    :param average_word_lengths: average length of words per message
    :type average_word_lengths: pd.Series
    '''    
    plt.hist(average_word_lengths,bins = int(max(average_word_lengths)))
    plt.xlim([0,50])
    plt.title('Average Word Length Histogram')
    plt.savefig('plots/average_word_lengths_hist.png',bbox_inches='tight',dpi=1000)

def plot_stop_dic(stop_dic,top_n=10):
    '''plots and saves the n top stop words' frequencies

    :param stop_dic: stop words (dict keys) and their frequencies (dict values)
    :type stop_dic: dictionary
    :param top_n: number of top stopwords to plot, defaults to 10
    :type top_n: int, optional
    '''    
    idx = np.argsort(list(stop_dic.values()))[::-1]
    words = np.array(list(stop_dic.keys()))
    freq = np.array(list(stop_dic.values()))
    plt.bar(words[idx[:top_n]], freq[idx[:top_n]])
    plt.title('Top 10 Stop Words Histogram')
    plt.savefig('plots/stop_dic_histogram.png',bbox_inches='tight',dpi=1000)

def plot_top_n_ngrams(ngrams,n=10):
    '''plots top n grams' frequencies

    :param ngrams: each ngram and its frequency in descending order
    :type ngrams: list of tuples
    :param n: number of top ngrams to show, defaults to 10
    :type n: int, optional
    '''    
    x,y = zip(*ngrams)
    plt.barh(list(x[:n]),list(y[:n]))

def preprocess_news(df):
    '''Preprocess message data to remove stop words and lemmatize 

    :param df: messages to preprocess in 'user_messages' column
    :type df: pd.DataFrame
    :return: corpus
    :rtype: list of words
    '''    
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
    '''plots and saves wordcloud

    :param wordcloud: wordcloud to be plotted 
    :type wordcloud: wordcloud.WordCloud
    '''    
    plt.imshow(wordcloud)
    plt.title('Word Cloud')
    plt.axis('off')
    plt.savefig('plots/word_cloud.png',bbox_inches='tight',dpi=1000)

def plot_unigrams(unigrams):
    '''plots and saves unigrams histogram

    :param unigrams: unigrams and frequencies
    :type unigrams: list of tuples
    '''    
    plot_top_n_ngrams(unigrams)
    plt.title('Top 10 Unigrams Histogram')
    plt.savefig('plots/unigrams.png',bbox_inches='tight',dpi=1000)

def plot_bigrams(bigrams):
    '''plots and saves bigrams histogram

    :param bigrams: bigrams and frequencies
    :type bigrams: list of tuples
    '''    
    plot_top_n_ngrams(bigrams)
    plt.title('Top 10 Bigrams Histogram')
    plt.savefig('plots/bigrams.png',bbox_inches='tight',dpi=1000)

def plot_trigrams(trigrams):
    '''plots and saves trigrams histogram

    :param trigrams: trigrams and frequencies
    :type trigrams: list of tuples
    '''     
    plot_top_n_ngrams(trigrams)
    plt.title('Top 10 Trigrams Histogram')
    plt.savefig('plots/trigrams.png',bbox_inches='tight',dpi=1000)

def get_plots(message_lengths,average_word_lengths,stop_dic,unigrams,bigrams,trigrams,wordcloud):
    '''Runs all plotting and saving functions, each in its own figure.
    All plots are saved in 'plots' folder

    :param message_lengths: each message's length in terms of words
    :type message_lengths: pd.Series
    :param average_word_length: each message's average word length
    :type average_word_length: pd.Series 
    :param stop_dic: frequencies (dict values) for each stopword (dict keys)
    :type stop_dic: collections.defaultdict
    :param unigrams: decending ordered, by frequency, list of tuples: (unigram, frequency)
    :type unigrams: list of tuples
    :param bigrams: decending ordered by frequency list of tuples: (bigram, frequency)
    :type bigrams: list of tuples
    :param trigrams: decending ordered by frequency list of tuples: (trigram, frequency)
    :type trigrams: list of tuples
    :param wordcloud: word cloud function, plot using plt.imshow
    :type wordcloud: worldcloud.WorldCloud
    '''    
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