# Used for preprocessing the affinn word list and create .pickle file of every word and its valence for beig positive or negative
# Used several language processing techniques which one can find in the code.
# I have also used wordnet to increase the corpus of word from 2477 to 5500 word valence dictionary.

import os
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.corpus import stopwords
import re
import threading


import pickle
pickle.dump(stemmer,open('stem.pkl','wb'),protocol=4)


path = r"C:\Users\shankul\Downloads\imm6010\AFINN"


with open(os.path.join(path,"AFINN-111.txt"),'r') as infile:
    text = {}
    for line in infile:
        ans = line.split()
        if len(ans) ==2:
            res = stemmer.stem(ans[0])
            text[res] = int(ans[1])
with open(os.path.join(path,"AFINN-96.txt"),'r') as infile:
    for line in infile:
        ans = line.split()
        if len(ans) ==2:
            res = stemmer.stem(ans[0])
            text[res] = int(ans[1])

os.chdir(r'C:\Users\shankul\Desktop\all')
df = pd.read_csv('test_imdb.csv')


stop = stopwords.words('english')




def preprocess(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+',' ',text.lower())
    text = [stemmer.stem(i) for i in text.split() if i not in stop]
    return text



def func(data,n):
    global sha
    token = preprocess(data)
    tot = 0
    for i in token:
        if i in text.keys():
            tot += text[i]
    if tot>= 0 and df.loc[n,"sentiment"]>= 0:
        sha=sha+1
    elif tot< 0 and df.loc[n,"sentiment"]< 0:
        sha=sha+1
    if n%500 == 0 and n!=0:
        print("Running accuracy- ",sha/n)



p=[]
sha = 0
th = []
for n,i in enumerate(df["review"]):
    p=threading.Thread(target = func,args = (i,n))
    p.start()
    th.append(p)
for i in th:
    i.join()


from nltk.corpus import wordnet as wn

for i in list(text):
    word_net = wn.synsets(i)
    for j in word_net:
        for k in j.lemma_names():
            val = stemmer.stem(k)
            if val not in text.keys():
                text[val] = text[i]
    
    


emoticons={':)':3,':(':-3,r'=(':-3,';)':2}
for em in list(emoticons):
    text[em] = emoticons[em]
           


import pickle
pickle.dump(text,open('afinn.pkl','wb'),protocol=4)

