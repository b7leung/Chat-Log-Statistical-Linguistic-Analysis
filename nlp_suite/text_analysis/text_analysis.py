import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from string import punctuation
from wordcloud import WordCloud, STOPWORDS
import itertools as it
import re



def frequency_info(Chat_dict, Username = None):
    '''[summary]

    :param Chat_dict: [description]
    :type Chat_dict: [type]
    :param Username: [description], defaults to None
    :type Username: [type], optional
    '''    
    
    frequency_list = [len(i) for i in Chat_dict.values()]
    
    df = pd.DataFrame({'Name': Chat_dict.keys(), 'freq' : frequency_list})
    df.sort_values(by = ['freq'], ascending = False, inplace = True)
    df.reset_index(drop = True, inplace = True)
    
    print('The most frequent speakers in this discord server are: ' + ', '.join(list(df['Name'][:5]))+
    '\n with more than ' + str(df['freq'][4]) + ' messages exchanged for each person')
    
    if Username != None:
        print('\n' + Username + ' is number {} frequent user in the server, with total {} messages'
              .format((np.where(df['Name'] == Username)[0][0]+1), df['freq'][np.where(df['Name'] == Username)[0][0]]))
    




def generate_word_cloud(chat_dict, max_words=200, width=500, height=500, background_color='white', title=""):
    '''[summary]

    :param chat_dict: [description]
    :type chat_dict: [type]
    :param max_words: [description], defaults to 200
    :type max_words: int, optional
    :param width: [description], defaults to 500
    :type width: int, optional
    :param height: [description], defaults to 500
    :type height: int, optional
    :param background_color: [description], defaults to 'white'
    :type background_color: str, optional
    :param title: [description], defaults to ""
    :type title: str, optional
    '''    
 
    def get_stop_words_wordcloud():
        """
        Helper function to identify words to be exclude from the word cloud
        """
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words.extend([c for c in punctuation])
        stop_words.extend(list(chat_dict.keys()))
        stop_words.extend([str(x) for x in np.arange(100)] +
                          ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"])
        return set(stop_words)
    
    # Clean up words by excluding stop words
    stop_words = get_stop_words_wordcloud()
    tokenized_ds = list(map(nltk.word_tokenize, list(it.chain(*list(chat_dict.values())))))
    tokenized_ds = [ (re.sub(r'[^\w\s]','',x)).lower() for ls in tokenized_ds for x in ls ]
    words = [word for word in tokenized_ds if word not in stop_words]
    
    # Generate word cloud
    wc = WordCloud(background_color=background_color, max_words=max_words, width=width, height=height)
    wc.generate(' '.join(words))
    
    # Plot the word cloud
    fig=plt.gcf()
    fig.set_size_inches(15,10)
    plt.imshow(wc)
    plt.axis('off')   
    plt.title(title, fontsize=25, fontweight="bold", pad=20)
    plt.show()


def plot_sentence_length_histogram(chat_dict, title = '', color = '#AFD5FA'):
    '''[summary]

    :param chat_dict: [description]
    :type chat_dict: [type]
    :param title: [description], defaults to ''
    :type title: str, optional
    :param color: [description], defaults to '#AFD5FA'
    :type color: str, optional
    '''    
        
    #break sentences into tokens
    tokenized_ds = list(map(nltk.word_tokenize, list(it.chain(*list(chat_dict.values())))))
    
    #count lengths
    lengths = []
    for i in tokenized_ds:
        if len(i) == 0:
            continue
        lengths.append(len(i))
    
    

    bins = np.arange(0,31,1)
    fig, ax = plt.subplots(figsize=(9, 5))
    _, bins, patches = plt.hist(np.clip(lengths, bins[0], bins[-1]),
                                density=False,
                                bins=bins, color=color, label='# messages')

    xlabels = bins[0:].astype(str)
    xlabels[-1] += '+'

    N_labels = len(xlabels)
    plt.xlim([0, 31])
    plt.xticks(np.arange(N_labels))
    ax.set_xticklabels(xlabels)
    
    

    plt.title(title, fontweight="bold")
    plt.xlabel('sentence lengths')
    plt.setp(patches, linewidth=0)
    plt.legend(loc='upper right')

    fig.tight_layout()

if __name__ == '__main__':
    pass