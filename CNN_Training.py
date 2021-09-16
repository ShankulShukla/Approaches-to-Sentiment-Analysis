# In this implementation we will be using CNNs for natural language processing. As CNNs are primarily famous for its usage in 2D images, in NLP for any text sequence we traet it as a one-dimensional image.
# One-dimensional CNNs can process local features such as  n-grams in text.
import tensorflow as tf
import pandas as pd
import numpy as np
import re
import os

# Importing the dataset of 50000 IMDB reviews.
df = pd.read_csv(os.getcwd()+r'\data\IMDB_Dataset.csv')


# for constructing a sort of dictionary which can be used to extract the word vectors.
from collections import Counter
count = Counter()

# Pre-processing the review text
# Removing html, hyperlinks
# Extracting emoticons and appending at the end of review, as emoticons in my analysis carry special meaning so not removing it
# Removing symbols, then converting to lower case
for i,text in enumerate(df['review']):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower()) 
    text = re.sub('[\W]+',' ',text.lower()) + ' '.join(emoticons).replace('-','')
    df.loc[i,"review"] = text 
    count.update(text.split())

# mapping target values to int
df.sentiment = [ 1 if each == "positive" else 0 for each in df.sentiment]

# one can directly import the word_to_int.pkl here which is in building classifier

word_count = sorted(count,key=count.get,reverse =True)

word_to_int = {word:i for i ,word in enumerate(word_count,1)}

seq_length = 200

mapped_reviews =[]
for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])


seq = np.zeros((len(mapped_reviews),seq_length),dtype = int)


for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    seq[i,-len(row):] = review_arr[-seq_length:]

x_train, x_test, y_train, y_test = seq[:35000], seq[35000:], df.loc[:34999,"sentiment"].values, df.loc[35000:,'sentiment'].values



#hyperparameters for the model
classes = 2
emb_size = 256
filter_sizes = [3,4,5]
num_filters = 128
n_epoch = 20
l_rate=0.001
b_size = 64
n_words = len(word_to_int)+1


tf.reset_default_graph()
tf.set_random_seed(21)


with tf.name_scope("inputs"):
    tf_x = tf.placeholder(dtype=tf.int64,shape=(None,seq_length),name = "tf_x")
    tf_y = tf.placeholder(dtype=tf.int32,shape=(None),name="tf_y")
    tf_keepprob = tf.placeholder(tf.float32,name="tf_keepprob")


with tf.name_scope("embeddings"):
    embedding = tf.Variable(tf.random_uniform(shape=(n_words,emb_size),minval=-1.0,maxval=1.0),name="embedding")
    embed_x = tf.nn.embedding_lookup(embedding, tf_x,name="embed_x")



#output for each filter size
output = []
for i in filter_sizes:
    with tf.name_scope("layer{}".format(i)):
        conv = tf.layers.conv1d(embed_x,filters = num_filters,kernel_size = i)
        act = tf.nn.relu(conv,"relu")
        pool = tf.layers.max_pooling1d(act,pool_size= seq_length-i+1 ,strides = 1,name="pool")
        output.append(pool)
outputs = tf.concat(output,-1)
flat_out = tf.contrib.layers.flatten(outputs)



with tf.name_scope("output"):
    drp = tf.nn.dropout(flat_out, tf_keepprob)
    logits = tf.layers.dense(drp, units=classes, activation=tf.nn.sigmoid)
    predict = tf.argmax(logits, 1, name='predict')
    y_hot = tf.one_hot(indices=tf_y,depth=classes,dtype = tf.int32)
    loss = tf.losses.softmax_cross_entropy(y_hot, logits=logits)
    optimizer  = tf.train.AdamOptimizer(l_rate).minimize(loss)
    cp = tf.equal(predict,tf.argmax(y_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(cp, tf.float32))

def batch_generator(x,y,batch_size):  
    for i in range(0,len(x),batch_size):
        if y is not None:
            yield x[i:i+batch_size], y[i:i+batch_size]
        else:
            yield x[i:i+batch_size]



saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # next line used for further training purpose that is loading the model again and training it
    # saver.restore(sess,tf.train.latest_checkpoint('./model_/'))
    for epoch in range(n_epoch):
        losses = []
        for batch_x, batch_y in batch_generator(x_train, y_train, b_size):
            feed = {tf_x: batch_x, tf_y: batch_y, tf_keepprob: 0.3}
            lossy, _ = sess.run([loss, optimizer], feed_dict=feed)
            losses.append(lossy)
        print('Loss in step {} is - {}'.format(epoch,np.average(losses)))
        if (epoch+1) % 10 == 0 :
            saver.save(sess,'model_/senti-{}.ckpt'.format(40+epoch))


preds = []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./model_/'))
    for batch_x, batch_y in batch_generator(x_test,y_test,b_size):
        # not using dropout in testing
        feed = {tf_x: batch_x, tf_y : batch_y, tf_keepprob: 1.0}
        acc = sess.run(accuracy,feed_dict=feed)
        preds.append(acc)

print("Accuracy- ",sum(preds)/len(preds))

