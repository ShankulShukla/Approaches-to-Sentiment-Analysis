#Multilayer LSTM based rnn model, due to lack of compute power I have loosely gone through the hyperparameter testing but the hyperparameters choosen provides best results. 

from collections import Counter
import pandas as pd
import numpy as np
import re


df1 = pd.read_csv(r"C:\Users\shankul\Desktop\all\train_imdb.csv")
df2 = pd.read_csv(r"C:\Users\shankul\Desktop\all\test_imdb.csv")


l = [df1,df2]
df= pd.concat(l,ignore_index=True)



print(df.tail())



count = Counter()


for i,text in enumerate(df['review']):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower()) 
    text = re.sub('[\W]+',' ',text.lower()) + ' '.join(emoticons).replace('-','')
    df.loc[i,"review"] = text 
    count.update(text.split())    


word_count = sorted(count,key=count.get,reverse =True)


word_to_int = {word:i for i ,word in enumerate(word_count,1)}



import pickle
pickle.dump(word_to_int,open('word_to_int.pkl','wb'),protocol=4)


int_to_word = {i:word for i ,word in enumerate(word_count,1)}


mapped_reviews =[]
for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])



seq_len = 200


seq = np.zeros((len(mapped_reviews),seq_len),dtype = int)


for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    seq[i,-len(row):] = review_arr[-seq_len:]


x_train, x_test, y_train, y_test = seq[:35000], seq[35000:], df.loc[:34999,"sentiment"].values, df.loc[35000:,'sentiment'].values
 

import tensorflow as tf


n_words = len(word_to_int)+1
lstm_size = 128
lstm_layers = 1
batch_size = 64
l_rate = 0.001
embed_size = 256
n_epoch = 40


tf.reset_default_graph()
tf.set_random_seed(123)
with tf.name_scope("inputs"):
    tf_x = tf.placeholder(dtype=tf.int32, shape=(None,seq_len) , name="tf_x")
    tf_y = tf.placeholder(dtype=tf.float32, shape=(None) , name="tf_y")
    tf_keepprob = tf.placeholder(tf.float32, name="tf_keepprob")


with tf.name_scope("embeddings"):
    embedding = tf.Variable(tf.random_uniform(shape=(n_words,embed_size),minval=-1,maxval=1),name="embedding")
    embed_x = tf.nn.embedding_lookup(embedding, tf_x,name="embed_x")


def lstm():
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    return tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob= tf_keepprob)
with tf.name_scope("rnn_layers"):
    cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(lstm_layers)])



with tf.name_scope("forward"):
    output_lstm, final_state = tf.nn.dynamic_rnn(cell, embed_x,dtype=tf.float32)


with tf.name_scope("output"):
    # giving the last output by -1
    logit = tf.layers.dense(inputs = output_lstm[:,-1],units=1,activation=None,name='logits')
    logits = tf.squeeze(logit,name="logits_squeeze")
    y_prob = tf.nn.sigmoid(logit,name='probabilities')



with tf.name_scope("optimize"):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y,logits=logits,name='cost'))
    optimizer = tf.train.AdamOptimizer(l_rate).minimize(cost,name='train_op')



def batch_generator(x,y,batch_size):  
    for i in range(0,len(x),batch_size):
        if y is not None:
            yield x[i:i+batch_size], y[i:i+batch_size]
        else:
            yield x[i:i+batch_size]

saver = tf.train.Saver()

#training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        losses = 0
        tot = 0
        for batch_x, batch_y in batch_generator(x_train, y_train, batch_size):
            feed = {tf_x: batch_x, tf_y: batch_y, tf_keepprob: 0.5}
            loss, _,state = sess.run([cost, optimizer,final_state], feed_dict=feed)
            losses =losses + loss
            tot = tot+1
        losses = losses / tot
        print("Epoch: {} , Train_loss: {}".format(epoch,losses))
        if (epoch+1) % 10 == 0 :
            saver.save(sess,'model/sentiment-{}.ckpt'.format(epoch))

preds = []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    for batch_x in batch_generator(x_test,None,batch_size):
        # not using dropout in testing
        feed = {tf_x: batch_x, tf_keepprob: 1.0}
        pred= sess.run([y_prob],feed_dict=feed)
        preds.append(pred)


result = np.concatenate(preds,axis=1)


result = result.reshape(-1,1)


round_ = np.around(result)


y_test = y_test.reshape(-1,1)


print(np.sum(round_ == y_test[:len(round_)])/len(round_))




