# bag words + TFIDF + multinomial naive bayes

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import os


def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower()) 
    text = re.sub('[\W]+',' ',text.lower()) + ' '.join(emoticons).replace('-','')
    return text



def vectorizer_unigram(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words = stopwords.words('english'))
    tfidf.fit(data)
    return tfidf

def fitting(x,y):
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.partial_fit(x,y,classes=np.array([0,1]))
    return clf



os.chdir(r'C:\Users\shankul\Desktop\all')
df = pd.read_csv('train_imdb.csv')



for i,data in enumerate(df['review']):
    df.loc[i,'review'] = preprocessing(data)



x, y = df['review'].values, df['sentiment'].values


vect = vectorizer_unigram(x)


transform = vect.transform(x)


clf = fitting(transform, y)


df1 = pd.read_csv('test_imdb.csv')


x_test, y_test = df1['review'].values, df1['sentiment'].values


transform_test = vect.transform(x_test)


pred = clf.predict(transform_test)


acc = (pred == y_test).sum()/len(y_test)



print(acc)



clf.partial_fit(transform_test,y_test)


pred = clf.predict(transform_test)
print((pred == y_test).sum()/len(y_test))


import pickle
pickle.dump(stopwords.words('english'),open('stop_words.pkl','wb'),protocol=4)
pickle.dump(clf,open('classifier.pkl','wb'),protocol=4)
pickle.dump(vect,open('vocab_.pkl','wb'),protocol = 4)




