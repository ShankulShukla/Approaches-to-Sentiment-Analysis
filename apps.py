# Creating web app using flask for making classifier into production

import re
import pickle
import os
import numpy as np
import tensorflow as tf

# change this directory where you have this repo folder
curr = r'C:\Users\shankul\Desktop\senti'
stop = pickle.load(open(os.path.join(curr,'building classifier','stop_words.pkl'),'rb'))
clf = pickle.load(open(os.path.join(curr,'building classifier','classifier.pkl'),'rb'))
vect = pickle.load(open(os.path.join(curr,'building classifier','vocab_.pkl'),'rb'))
text = pickle.load(open(os.path.join(curr,'building classifier','afinn.pkl'),'rb'))
stemmer = pickle.load(open(os.path.join(curr,'building classifier','stem.pkl'),'rb'))
word_to_int = pickle.load(open(os.path.join(curr,'building classifier','word_to_int.pkl'),'rb'))


# Bag of word using multinomial naive bayes

def preprocessing(text):
    # Removing the tags
    text = re.sub('<[^>]*>', '', text)
    # Extracting the emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')
    return text

def classify1(X):
    prediction = {1:'Positive',0:'Negative'}
    X = vect.transform([X])
    pred = clf.predict(X)[0]
    probability = np.max(clf.predict_proba(X))
    return prediction[pred], probability


# AFINN word valence

def preprocess(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+',' ',text.lower())
    text = [stemmer.stem(i) for i in text.split() if i not in stop]
    return text


def classify(review):
    token = preprocess(review)
    tot = 0
    for i in token:
        if i in text.keys():
            tot += text[i]
    if tot>= 0 :
        return "Positive", tot
    elif tot< 0 :
        return "Negative", tot

# LSTM neural network

seq_len = 200 # as used in training
batch_size = 64
n_words = len(word_to_int)+1
embed_size = 256
lstm_layers = 1
lstm_size = 128


def classify2(review):
    review = preprocessing(review)
    mapped_review = []
    for word in review.split():
        if word in word_to_int.keys():
            mapped_review.append(word_to_int[word])
        else:
            mapped_review.append(0)
    seq = np.zeros((1,seq_len),dtype = int)
    review_arr = np.array(mapped_review)
    seq[0,-len(mapped_review):] = review_arr[-seq_len:]
    with tf.Session() as sess:
        # Change to that directory where your saved model is
        saver = tf.train.import_meta_graph(r'C:\Users\shankul\Desktop\senti\building classifier\model\sentiment-19.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(r'C:\Users\shankul\Desktop\senti\building classifier\model/'))
        # not using dropout in testing
        graph = tf.get_default_graph()
        tf_x = graph.get_tensor_by_name("inputs/tf_x:0")
        tf_keepprob = graph.get_tensor_by_name("inputs/tf_keepprob:0")
        y_prob = graph.get_tensor_by_name("output/probabilities:0")
        feed = {tf_x: seq, tf_keepprob: 1.0}
        pred = sess.run([y_prob],feed_dict=feed)
    return pred



from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/check1',methods = ['Post'])
def result():
    review_ = request.form['review']
    pred, score = classify(review_)
    return render_template('resultpage.html',review = review_,prediction = pred, data = "score", probability = str(score))

@app.route('/check2',methods = ['Post'])
def result1():
    review_ =request.form['review']
    pred, prob = classify1(review_)
    return render_template('resultpage.html',review = review_,prediction = pred, data = "accuracy", probability = str(round(prob*100,3))+"%")

@app.route('/check3',methods = ['Post'])
def result2():
    review_ = request.form['review']
    pred= classify2(review_)[0][0][0]
    if round(pred) == 0:
        res = "Negative"
    else:
        res = "Positive"
    return render_template('resultpage.html',review = review_,prediction = res, data = 'accuracy', probability = str(pred*100)+"%")

if __name__ == '__main__':
    app.run(debug=True)
