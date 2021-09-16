

import numpy as np
import re
import os
import pickle

clf_unigram = pickle.load(open(os.path.join(os.getcwd(), 'models', 'NBclassifier_unigram.pkl'), 'rb'))
clf_bigram = pickle.load(open(os.path.join(os.getcwd(), 'models', 'NBclassifier_bigram.pkl'), 'rb'))
vect_unigram = pickle.load(open(os.path.join(os.getcwd(), 'models', 'tfidf_unigram.pkl'), 'rb'))
vect_bigram = pickle.load(open(os.path.join(os.getcwd(), 'models', 'tfidf_bigram.pkl'), 'rb'))


# Pre-processing the review text before tokenizing using TF-IDF
# Removing html, hyperlinks
# Removing symbols
# Extracting emoticons and appending at the end of review, as emoticons in my analysis carry special meaning so not removing it
def preprocessing(text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', text)
    newtext = re.sub(r'http\S+', ' ', cleantext)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', newtext)
    text = re.sub('[\W]+', ' ', newtext.lower()) + ' '.join(emoticons).replace('-', '')
    return text


# Expanding contractions before pre-processing review text
# Focusing mainly on word "not" as I have used it to create reverse associations among phrases, explained further
contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not",
                       "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not",
                       "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would",
                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                       "let's": "let us", "ma'am": "madam",
                       "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have",
                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                       "needn't've": "need not have", "o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have",
                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is",
                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have", "so's": "so as",
                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would",
                       "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                       "they've": "they have", "to've": "to have",
                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                       "we'll've": "we will have", "we're": "we are",
                       "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are",
                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                       "where'd": "where did", "where's": "where is",
                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                       "who've": "who have",
                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                       "won't've": "will not have",
                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                       "y'all": "you all",
                       "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                       "y'all've": "you all have",
                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                       "you'll've": "you will have",
                       "you're": "you are", "you've": "you have"}


# List to identify negation words in review
negation = ['no', 'not', 'none', 'never', 'hardly', 'scarcely', 'barely', 'donot']


# stop_words for identifying stop words and removing it from review
# As stop words generally do not pertains any special meaning not considering in this implementation
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
              "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
              "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "of", "at", "by", "for", "with",
              "between", "into", "during", "before", "after", "above", "below", "to", "from",
              "up", "down", "in", "out", "on", "off", "over", "under", "again", "once", "here", "there",
              "when", "where", "why", "how", "all", "any", "both", "each", "other", "such", "own", "same", "too",
              "very", "s", "t", "can", "will", "just", "don", "should", "now"]


# Removing stop word from review text
def remove_stop_words(text):
    text = ' '.join(word for word in text.split()
                    if word.lower() not in stop_words)
    return text


# Pre-processing the review using above defined contraction mapping dictionary
def contractionmap(text):
    for n in contraction_mapping.keys():
        text = text.replace(n, contraction_mapping[n])
    return text


# In this function, I will prepend the prefix "not" to every word after a token of logical negation(negation list above) until the next punctuation mark.
# EX- "I did not like this movie", in this example classifier will interpret "like" as a different feature(positive feature) ignoring its association with negation "not"
# After adding negation prefix, "like" becomes "notlike" will thus occur more often in negative document and act as cues for negative sentiment
# Similarly, words like "notbad" will acquire positive associations with positive sentiment
# Returning pre-processed review text
def negationprefix(text):
    text = remove_stop_words(text)
    token = text.split()
    negationFlag = False
    processedtoken = []
    # Punctuation to stop
    punctuation = re.compile(r'[.,!?;]')
    for i in token:
        if i in negation:
            negationFlag = True
            processedtoken.append(i)
        else:
            if negationFlag:
                processedtoken.append("not" + i)
            else:
                processedtoken.append(i)
        if punctuation.search(i):
            negationFlag = False

    newText = ' '.join(processedtoken)
    return preprocessing(newText)

def classify(text, ngram):
    prediction = {1: 'Positive', 0: 'Negative'}
    text = contractionmap(text)
    text = negationprefix(text)
    if ngram == 1:
        vect = vect_unigram
        clf = clf_unigram
    else:
        vect = vect_bigram
        clf = clf_bigram
    X = vect.transform([text])
    pred = clf.predict(X)[0]
    probability = np.max(clf.predict_proba(X))
    return prediction[pred], probability

