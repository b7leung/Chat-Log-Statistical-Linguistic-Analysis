'''
Preprocess data obtained from https://www.kaggle.com/jef1056/discord-data
'''

import os
from pathlib import Path
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Preprocess discord chat data')
parser.add_argument('--datapath', type=Path, required=True)
parser.add_argument('--outpath', type=Path, default=Path())
parser.add_argument('--fname', type=str, default='user_data.csv')
parser.add_argument('--minchats', type=int, default=10)
args = parser.parse_args()

def read_file(file):
    '''
    '''
    x = open(file,'r', encoding = 'utf-8') 
    y = x.read().replace('\\n', '\n').replace('\t', '\n')    #my dumb way to split lines, replace this for better method
    content = y.splitlines() 
    return content

def Sort_by_user(chat):
    '''
    '''
    Sort_by_User = {}
    for i in range(len(chat)):
        if ':' in chat[i] and chat[i][0] != ':':     #check if switch user
            name = chat[i].split(':', 1)[0]
            if name not in Sort_by_User:             #check if it's new user
                Sort_by_User[name] = [chat[i].split(':', 1)[1]]
            else:
                Sort_by_User.update(name = [Sort_by_User[name].append(chat[i].split(':', 1)[1].lstrip())])  #add line under user key
        else:
            Sort_by_User.update(name = [Sort_by_User[name].append(chat[i])])   #if current user continue next line
            
    del Sort_by_User['name']
    return Sort_by_User


if __name__ == '__main__':
    files = [os.path.join(dirname, filename) for dirname, _, filenames in os.walk(args.datapath) \
            for filename in filenames if filename.endswith('.txt')]

    chats = []
    for f in files:
        chats.extend(read_file(f))

    chats = Sort_by_user(chats)
    chats = pd.DataFrame(chats.items(), columns=['User', 'Chats'])
    chats = chats[chats['Chats'].map(len) >= args.minchats]

    chats.to_csv(args.outpath / args.fname, index=False)
