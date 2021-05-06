import os
import pprint
import re
from collections import defaultdict
import glob


def clean_line(line):

    cleaned_line = line
    cleaned_line = cleaned_line.replace('\\n', ' ') # 
    cleaned_line = cleaned_line.replace('\n', ' ') # 
    cleaned_line = cleaned_line.split('\t') # users in the same line are split by tabs

    return cleaned_line


def process_discord_data(chat_log_paths, min_line_length=0):
    
    #chat_log_paths = glob.glob(os.path.join(data_dir, "*{}*".format(channel_id)))
    channel_messages = defaultdict(list)
    
    for chat_log_path in chat_log_paths:
        #https://stackoverflow.com/questions/21129020/how-to-fix-unicodedecodeerror-ascii-codec-cant-decode-byte
        with open (chat_log_path, 'r', encoding='utf-8') as f:
            f = f.read()
            f = f.split('\n')
            
        for line_num in range(len(f)):
            for message in clean_line(f[line_num]):
                split_msg = message.split(": ")
                user = split_msg[0].replace(" ", "")
                message_text = ": ".join(split_msg[1:]) # handle other cases
                if len(message_text.split(' ')) > min_line_length:
                    channel_messages[user].append(message_text)

    channel_messages = dict(channel_messages)

    message_counts = sorted([[user, len(channel_messages[user])] for user in channel_messages], key=lambda x: x[1], reverse=True)

    return channel_messages, message_counts

