# In this implementation we are using RNN with Long Short Term Memory Units.
# These units make sure that the hidden state vector is able to maintain long term dependencies in the text.
# I am using word vector representation as input to RNN, I have to transform text information to these vectors as word similar in context, meaning, and semantics reside's in relatively the same area in the vector space since they both have similar definitions and are both used in similar contexts.
# In this implementation, I will be training my own word vector representation from the corpora of word available in the imdb dataset.

from collections import Counter
import pandas as pd
import numpy as np
import re
import os


# Importing the dataset of 50000 IMDB reviews.
df = pd.read_csv(os.getcwd()+r'\data\IMDB_Dataset.csv')


# To create set of unique words and their counts which will be used to create the word vectors of words in review.
count = Counter()

# Pre-processing the review text
# Removing html, hyperlinks
# Extracting emoticons and appending at the end of review, as emoticons in my analysis carry special meaning so not removing it
# Removing symbols, then converting to lower case
for i,text in enumerate(df['review']):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    df.loc[i,"review"] = text 
    count.update(text.split())    

# create mapping
# map each unique word to an integer
word_count = sorted(count,key=count.get,reverse =True)

# this will be used t convert the entire text of a review into a list of numbers
word_to_int = {word:i for i, word in enumerate(word_count, 1)}

# exporting word vector embedding for deployment/ testing
import pickle
pickle.dump(word_to_int,open('models/RNN_word_to_int.pkl','wb'),protocol=4)

# converting sequence of words into sequence of integers
mapped_reviews =[]
for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])

# defining so that all review sequence should have same length
seq_len = 200

seq = np.zeros((len(mapped_reviews),seq_len),dtype = int)

# pad or truncate review w.r.t sequence length
for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    seq[i,-len(row):] = review_arr[-seq_len:]

# mapping target values to int
df.sentiment = [ 1 if each == "positive" else 0 for each in df.sentiment]

# splitting the dataset into training and testing data
x_train, x_test, y_train, y_test = seq[:35000], seq[35000:], df.loc[:34999,"sentiment"].values, df.loc[35000:,'sentiment'].values
 

import tensorflow as tf

# hyper-parameters for LSTM-RNN model
n_words = len(word_to_int)+1
lstm_size = 128
lstm_layers = 1
batch_size = 64
l_rate = 0.001
embed_size = 256
n_epoch = 40


tf.reset_default_graph()
tf.set_random_seed(123)

# defining placeholder for the model
with tf.name_scope("inputs"):
    tf_x = tf.placeholder(dtype=tf.int32, shape=(None, seq_len), name="tf_x")
    tf_y = tf.placeholder(dtype=tf.float32, shape=(None), name="tf_y")
    tf_keepprob = tf.placeholder(tf.float32, name="tf_keepprob")

# we initilise our embedding randomly with float values between 1 and -1
# then we use lookup function to loop up the row in embedding matrix for each element in the placeholder tf_x
with tf.name_scope("embeddings"):
    embedding = tf.Variable(tf.random_uniform(shape=(n_words, embed_size), minval=-1, maxval=1), name="embedding")
    embed_x = tf.nn.embedding_lookup(embedding, tf_x, name="embed_x")

# Function to create a single layer of LSTM
def lstm():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob= tf_keepprob)

# Can create multiple stacked layers based on lstm_layers input parameter
# Stacking these cells helps the model retain more long term dependence information, but also introduces more parameters, more training time and also model might overfit(depend on number of input samples)
with tf.name_scope("rnn_layers"):
    cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(lstm_layers)])

# tf.nn.dynamic_rnn() performs unrolling the whole network and creating a pathway for the data to flow through the RNN graph
with tf.name_scope("forward"):
    output_lstm, final_state = tf.nn.dynamic_rnn(cell, embed_x,dtype=tf.float32)


with tf.name_scope("output"):
    # giving the last output by -1
    logit = tf.layers.dense(inputs=output_lstm[:, -1], units=1, activation=None, name='logits')
    logits = tf.squeeze(logit, name="logits_squeeze")
    y_prob = tf.nn.sigmoid(logit, name='probabilities')

# defining optimizer to train the model parameters
with tf.name_scope("optimize"):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y,logits=logits,name='cost'))
    optimizer = tf.train.AdamOptimizer(l_rate).minimize(cost,name='train_op')


# breaks the dataset into mini-batches, returns a generator to iterate through these chunks
def batch_generator(x, y, batch_size):
    for i in range(0,len(x),batch_size):
        if y is not None:
            yield x[i:i+batch_size], y[i:i+batch_size]
        else:
            yield x[i:i+batch_size]

saver = tf.train.Saver()

# Training LSTM_RNN model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        losses = 0
        tot = 0
        for batch_x, batch_y in batch_generator(x_train, y_train, batch_size):
            feed = {tf_x: batch_x, tf_y: batch_y, tf_keepprob: 0.5}
            loss, _, state = sess.run([cost, optimizer,final_state], feed_dict=feed)
            losses = losses + loss
            tot = tot+1
        losses = losses / tot
        print("Epoch: {} , Train_loss: {}".format(epoch, losses))
        if (epoch+1) % 10 == 0:
            saver.save(sess,'models/sentiment-{}.ckpt'.format(epoch))

# Testing and prediction on testing data
preds = []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./models/'))
    for batch_x in batch_generator(x_test,None,batch_size):
        # not using dropout in testing
        feed = {tf_x: batch_x, tf_keepprob: 1.0}
        pred = sess.run([y_prob], feed_dict=feed)
        preds.append(pred)


result = np.concatenate(preds,axis=1)

result = result.reshape(-1,1)

round_ = np.around(result)
y_test = y_test.reshape(-1,1)

accuracy = np.sum(round_ == y_test[:len(round_)])/len(round_)

print("Accuracy on 10000 testing data is- ",accuracy)



