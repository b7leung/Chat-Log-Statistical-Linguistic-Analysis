import os


def read_file(file):
    '''
    read text file into a list
    '''
    x = open(file,'r', encoding = 'utf-8') 
    y = x.read().replace('\\n', '\n').replace('\t', '\n')   
    content = y.splitlines() 
    return content

def Sort_by_user(chat):
    '''
    Sort the chatlog by user
    
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
    files = [os.path.join(dirname, filename) for dirname, _, filenames in os.walk("discord-v1") \
            for filename in filenames if filename.endswith('.txt')]