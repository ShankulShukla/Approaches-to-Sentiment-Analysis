# AFINN is a list of English words rated for valence with an integer
# between minus five (negative) and plus five (positive).
# In this implementation I am using AFINN-111, which is the newest version with 2477 words.

import pandas as pd
import os
import re


# Importing AFINN lexicon list, and created a dictionary mapping of word with its lexicon score
with open(os.path.join(os.getcwd()+r'\data',"AFINN-111.txt"),'r') as infile:
    wordkey = {}
    for line in infile:
        ans = line.split()
        if len(ans) == 2:
            res = ans[0]
            wordkey[res] = int(ans[1])


# Importing the dataset of 50000 IMDB reviews.
df = pd.read_csv(os.getcwd()+r'\data\IMDB_Dataset.csv')

# Pre-processing the reviews.
# In each review, I removed all html tags and website link. No symbols or case is are removed as they represent
# special meaning, utilised further
def cleanhtml(text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', text)
    newtext = re.sub(r'http\S+', ' ', cleantext)
    return newtext

# Applying the text pre-processing
df['review']=df['review'].apply(cleanhtml)

# In this dataset, sentiment are often expressed in terms of emoticons. So, giving lexicon weights to most used emoticons
emoticons = {':)':3,':(':-3,r'=(':-3,';)':2}
for em in list(emoticons):
    wordkey[em] = emoticons[em]


# Degree modifiers to alter sentiment intensity.

# Positive intensifier: Ex- "very" when used "good", intensify its sentiment i.e., "very good" has more positive sentiment then "good"
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

# Negative intensifier: Ex- "very" when used "bad", intensify its sentiment i.e., "very bad" has more negative sentiment then "bad"
negative_intensifier = [ "almost"  , "barely"  , "hardly"  ,
     "less"  ,'least', "little"  , "marginally"  ,
     "occasionally"  , "partly"  , "scarcely"  , "slight"  , "slightly"  , "somewhat"  ]


# Negative contractions and negations are used to negate ideas.
# Example - "good" have positive connotation to it but "not good" as strong negative connotation, so a "not" negations changes idea of being good.
negation = ["aren't", "can't", 'cannot', "couldn't", "daren't", "didn't", 
            "doesn't", "don't", "hasn't", "haven't", "hadn't", "isn't", "mayn't", "mightn't", 
            "mustn't", "needn't", "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't", 
            'no', 'not', 'none', 'no one', 'nobody', 'nothing', 'neither', 'nowhere', 'never', 
            'hardly', 'scarcely', 'barely', 'doesn’t', 'isn’t', 'wasn’t', 'shouldn’t', 'wouldn’t', 'couldn’t', 'won’t', 
            'can’t', 'don’t']


# Not used in the current implementation, but these diagrams represent major positive or negative connotation in the dataset
positive_ngram = ["great movie", "good movie", "well done","great film","best movie","best film", "recommend movie", "recommend film"] 
negative_ngram = ["worst movie", "worst film", "waste time", "bad movie"]


# As AFINN only have 2462 word mapping, so to add extended corpora of words for the task, I used NLTK synset instances.
# It is a groupings of synonymous words that express the same concept, so for each word in AFINN word list I look up synonyms in synset
# and update the word valence mapping with new synonym with the valence value of its parent, if synonym is not already in the word valence mapping created
from nltk.corpus import wordnet as wn
for i in list(wordkey):
    word_net = wn.synsets(i)
    for j in word_net:
        for k in j.lemma_names():
            if k not in wordkey.keys():
                wordkey[k] = wordkey[i]



# For stemming the word if not present in word score dictionary created and then querying the dictionary for word stem
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# Remove symbols from the unigrams of the sentiment.
def removesymbol(word):
    word = re.sub(r'[^\w]', '', word)
    return word

# All these constants are fixed based on extensive trial on the dataset

# Identifying exclamations in the review as people conventionally do use punctuation to signal increased sentiment intensity
# In this implementation we considering exclamation symbol
# EX - great!!! have more positive intensity than great
# Whenever I encounter any exclamation, I increase the score of the word associated with exclamation by a factor of exclamationfactor
exclamationfactor = 1.7

# Identifying words are all in capital letters, people do use of word-shape to signal emphasis
# EX - "movie was BAD" have more negative intensity than "movie was bad"
# Whenever I encounter any all caps word, I increase the score of the word by a factor of capsfactor
capsfactor = 2

# Negation are used to negate ideas
# If any word in the review is in negation list defined above, we use negationFlag to mark its occurrence and use the negationfactor to reverse the score value of next available item.
negationfactor = -1

# If any word in the review is in positive_intensifier list defined above, we use positiveIntensifierFlag to mark its occurrence and multiply the score value of next available item by a factor of positiveintensifactor
positiveintensifactor = 1.5

# If any word in the review is in negative_intensifier list defined above, we use negativeIntensifierFlag to mark its occurrence and multiply the score value of next available item by a factor of negativeintensifactor
negativeintensifactor = -1.5

# Used to determine the accuracy of the model
rightclassify = 0 

# Applying sentiment analysis algorithm based on AFINN word list.
for i in range(len(df)):
    # for each review
    unigram = df.iloc[i,0].split()
    scorearray = []
    negationFlag = False
    positiveIntensifierFlag = False
    negativeIntensifierFlag= False
    for j in range(len(unigram)):
        # checking for emoticons
        if unigram[j] in emoticons.keys():
            scorearray.append(wordkey[unigram[j]])
        else:
            # exclamation point - emphasis
            if unigram[j].count('!')>0:
                temp = removesymbol(unigram[j]).lower()
                if temp in wordkey.keys():
                    score = wordkey[temp] * exclamationfactor * (capsfactor if  unigram[j].isupper() else 1)
                    scorearray.append(score)
                elif stemmer.stem(temp) in wordkey.keys():
                    score = wordkey[stemmer.stem(temp)] * exclamationfactor * (capsfactor if  unigram[j].isupper() else 1)
                    scorearray.append(score)
                else:
                    # Applying the factor on last visited valence word
                    if len(scorearray) > 1:
                        scorearray.append( scorearray[-1]  * exclamationfactor)
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
                        score = wordkey[temp]*(capsfactor if  unigram[j].isupper() else 1)*(negationfactor if  negationFlag else 1)\
                                *(positiveintensifactor if  positiveIntensifierFlag else 1)*(negativeintensifactor if  negativeIntensifierFlag else 1)
                        negationFlag = False
                        positiveIntensifierFlag = False
                        negativeIntensifierFlag = False
                        scorearray.append(score)
                    # If word not found in word list mapping, check for its stem or root word
                    elif stemmer.stem(temp) in wordkey.keys():
                        score = wordkey[stemmer.stem(temp)]*(capsfactor if  unigram[j].isupper() else 1)*\
                                (negationfactor if  negationFlag else 1)*(positiveintensifactor if  positiveIntensifierFlag else 1)*\
                                (negativeintensifactor if  negativeIntensifierFlag else 1)
                        negationFlag = False
                        positiveIntensifierFlag = False
                        negativeIntensifierFlag = False
                        scorearray.append(score)
    # Threshold for classification
    if len(scorearray) > 0:
        tot = sum(scorearray)/len(scorearray)
        if (tot >= 0 and df.iloc[i,1] == 'positive') or (tot < 0 and df.iloc[i,1] == 'negative'):
            rightclassify += 1

print("Accuracy on 50000 data is-",(rightclassify/50000)*100)

# Exporting the lexicon model for testing/deployment
import pickle
pickle.dump(wordkey,open('models/afinnModel.pkl','wb'),protocol=4)


