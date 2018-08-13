# Combining all text files into csv file which you will find after extracting the tar file of aclImdb.

import os
import pandas as pd
import numpy as np

# Change directory to where you have stored the text files
os.chdir(r'C:\Users\shankul\Downloads\aclImdb_v1\aclImdb')


train = pd.DataFrame()
test = pd.DataFrame()
label = {'pos':1, 'neg':0}


for file in ['test','train']:
    for data in ['pos','neg']:
        path = os.path.join(os.getcwd(),file,data)
        for text in os.listdir(path):
            with open(os.path.join(path,text),'r',encoding='utf-8') as infile:
                txt = infile.read()
            if file == 'test':
                test = test.append([[txt,label[data]]],ignore_index=True)
            else:
                train = train.append([[txt,label[data]]],ignore_index=True)
                
test.columns = ['review', 'sentiment']
train.columns = ['review', 'sentiment']
                

train = train.reindex(np.random.permutation(train.index))
test = test.reindex(np.random.permutation(test.index))


train.to_csv('train_imdb.csv',index=False,encoding='utf-8')
test.to_csv('test_imdb.csv',index=False,encoding='utf-8')






