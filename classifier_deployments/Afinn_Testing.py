# AFINN deployment file
# To understand the lexicon based pipeline, refer Afinn_Training.py

import os
import re
import pickle
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# Importing AFINN lexicon model
wordkey = pickle.load(open(os.path.join(os.getcwd(), 'models', 'afinnModel.pkl'), 'rb'))

positive_intensifier = [
   "absolutely"  , "awfully"  ,
     "completely"  , "considerably"  ,
     "decidedly"  , "deeply"  , "enormously"  ,
     "entirely"  , "especially"  , "exceptionally"  , "extremely"  ,
     'fairly',   "fully"  ,
     "greatly"  , "highly"   , "incredibly"  , "intensely"  ,
     "majorly"  , "more"  , "most"  ,"much"  , "positively"  ,
     "purely"  , "quite"  , "really"  , "remarkably"  ,
     "so"  , "substantially"  ,'strongly', 'too'
     "thoroughly"  , "totally"  , "tremendously"  ,
      "unbelievably"  , "unusually"  , "utterly"  ,
     "very"
]

negative_intensifier = [ "almost"  , "barely"  , "hardly"  ,
     "less"  ,'least', "little"  , "marginally"  ,
     "occasionally"  , "partly"  , "scarcely"  , "slight"  , "slightly"  , "somewhat"  ]


negation = ["aren't", "can't", 'cannot', "couldn't", "daren't", "didn't",
            "doesn't", "don't", "hasn't", "haven't", "hadn't", "isn't", "mayn't", "mightn't",
            "mustn't", "needn't", "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
            'no', 'not', 'none', 'no one', 'nobody', 'nothing', 'neither', 'nowhere', 'never',
            'hardly', 'scarcely', 'barely', 'doesn’t', 'isn’t', 'wasn’t', 'shouldn’t', 'wouldn’t', 'couldn’t', 'won’t',
            'can’t', 'don’t']

emoticons = {':)':3,':(':-3,r'=(':-3,';)':2}

exclamationfactor = 1.7

capsfactor = 2

negationfactor = -1

positiveintensifactor = 1.5

negativeintensifactor = -1.5

def preprocess(text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', text)
    newtext = re.sub(r'http\S+', ' ', cleantext)
    return newtext

def removesymbol(word):
    word = re.sub(r'[^\w]', '', word)
    return word

def classify(review):
    text = preprocess(review)
    unigram = text.split()
    scorearray = []
    negationFlag = False
    positiveIntensifierFlag = False
    negativeIntensifierFlag = False
    for j in range(len(unigram)):
        # checking for emoticons
        if unigram[j] in emoticons.keys():
            scorearray.append(wordkey[unigram[j]])
        else:
            # exclamation point - emphasis
            if unigram[j].count('!') > 0:
                temp = removesymbol(unigram[j]).lower()
                if temp in wordkey.keys():
                    score = wordkey[temp] * exclamationfactor * (capsfactor if unigram[j].isupper() else 1)
                    scorearray.append(score)
                elif stemmer.stem(temp) in wordkey.keys():
                    score = wordkey[stemmer.stem(temp)] * exclamationfactor * (capsfactor if unigram[j].isupper() else 1)
                    scorearray.append(score)
                else:
                    # Applying the factor on last visited valence word
                    if len(scorearray) > 1:
                        scorearray.append(scorearray[-1] * exclamationfactor)
            else:
                temp = unigram[j].lower()
                # checking for negative contradiction, negation, positive and negative intensifiers
                if temp in negation:
                    negationFlag = True
                elif temp in positive_intensifier:
                    positiveIntensifierFlag = True
                elif temp in negative_intensifier:
                    negativeIntensifierFlag = True
                else:
                    temp = removesymbol(unigram[j]).lower()
                    if temp in wordkey.keys():
                        score = wordkey[temp] * (capsfactor if unigram[j].isupper() else 1) * (
                            negationfactor if negationFlag else 1) \
                                * (positiveintensifactor if positiveIntensifierFlag else 1) * (
                                    negativeintensifactor if negativeIntensifierFlag else 1)
                        negationFlag = False
                        positiveIntensifierFlag = False
                        negativeIntensifierFlag = False
                        scorearray.append(score)
                    # If word not found in word list mapping, check for its stem or root word
                    elif stemmer.stem(temp) in wordkey.keys():
                        score = wordkey[stemmer.stem(temp)] * (capsfactor if unigram[j].isupper() else 1) * \
                                (negationfactor if negationFlag else 1) * (
                                    positiveintensifactor if positiveIntensifierFlag else 1) * \
                                (negativeintensifactor if negativeIntensifierFlag else 1)
                        negationFlag = False
                        positiveIntensifierFlag = False
                        negativeIntensifierFlag = False
                        scorearray.append(score)
    # Threshold for classification
    if len(scorearray) > 0:
        tot = sum(scorearray) / len(scorearray)
        if tot >= 0:
            return "positive", tot
        else:
            return "negative", tot







