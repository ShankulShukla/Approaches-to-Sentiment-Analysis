# LSTM neural network
import tensorflow as tf
import numpy as np
import os
import re


# Importing RNN word to int mapping
import pickle

word_to_int = pickle.load(open(os.path.join(os.getcwd(), 'models', 'RNN_word_to_int.pkl'), 'rb'))

seq_len = 200 # as used in training
batch_size = 64
n_words = len(word_to_int)+1
embed_size = 256
lstm_layers = 1
lstm_size = 128

def preprocessing(text):
    # Removing the tags
    text = re.sub('<[^>]*>', '', text)
    # Extracting the emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')
    return text


def classify(review):
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

        saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), "models", "CNN","senti-59.ckpt.meta"))
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(os.getcwd(), "models", "CNN")))
        # not using dropout in testing
        graph = tf.get_default_graph()
        tf_x = graph.get_tensor_by_name("inputs/tf_x:0")
        tf_keepprob = graph.get_tensor_by_name("inputs/tf_keepprob:0")
        y_prob = graph.get_tensor_by_name("output/predict:0")
        feed = {tf_x: seq, tf_keepprob: 1.0}
        pred = sess.run([y_prob],feed_dict=feed)
    pred = pred[0][0]
    if round(pred) == 0:
        res = "Negative"
    else:
        res = "Positive"
    # low probability means it is a mixed review
    return res, pred

