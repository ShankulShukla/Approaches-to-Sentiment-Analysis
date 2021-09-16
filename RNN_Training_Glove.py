# In this implementation we are using RNN with Long Short Term Memory Units.
# These units make sure that the hidden state vector is able to maintain long term dependencies in the text.
# I am using word vector representation as input to RNN, I have to transform text information to these vectors as word similar in context, meaning, and semantics reside's in relatively the same area in the vector space since they both have similar definitions and are both used in similar contexts.
# In this implementation, I am using a pre-trained word vector i.e., GloVe it much a much more manageable matrix then word2vec due its size. The matrix will contain 400,000 word vectors, each with a dimensionality of 50.

import numpy as np
import re
import os
import pandas as pd

# Importing GloVe and creating, vocabulary list and the embedding matrix.
with open(os.getcwd()+r"\data\glove.6B.50d.txt", encoding="utf8") as infile:
    embeddings_vector = []
    word_list = []
    for line in infile:
        values = line.split()
        word_list.append(values[0])
        emb = np.asarray(values[1:], dtype='float32')
        embeddings_vector.append(emb)
embeddings_vector = np.array(embeddings_vector)

# Importing the dataset of 50000 IMDB reviews.
df = pd.read_csv(os.getcwd()+r'\data\IMDB_Dataset.csv')

# Expanding contractions before pre-processing review text
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


# Pre-processing the review text
# Removing html, hyperlinks
# Removing symbols, then converting to lower case
def preprocessing(text):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', text)
    newtext = re.sub(r'http\S+', ' ', cleantext)
    text = re.sub('[\W]+',' ',newtext.lower())
    return text

# stop_words for identifying stop words and removing it from review
# As stop words generally do not pertains any special meaning not considering in this implementation
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", 
              "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", 
              "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", 
              "a", "an", "the", "and", "but", "if", "or", "because", "as", "of", "at", "by", "for", "with", 
               "between", "into", "during", "before", "after", "above", "below", "to", "from", 
              "up", "down", "in", "out", "on", "off", "over", "under", "again",  "once", "here", "there", 
              "when", "where", "why", "how", "all", "any", "both", "each", "other","such", "own", "same",  "too", "very", "s", "t", "can", "will", "just", "don", "should"
              , "now"]

# Removing stop words from review text
def remove_stop_words(text):
    text = ' '.join(word for word in text.split() 
                      if word.lower() not in stop_words)
    return text

# Pre-processing the review using above defined contraction mapping dictionary
def contractionmap(text):
    for n in contraction_mapping.keys():
        text = text.replace(n, contraction_mapping[n])
    return text

# Encapsulating pre-processing
def processReview(text):
    text = contractionmap(text)
    text = remove_stop_words(text)
    return preprocessing(text)

# Applying pre-processing to reviews
df['review']=df['review'].apply(processReview)

# From the histogram of average number of words per review, can safely say that most reviews will fall under 300 words.
avg_seq_len = 300

# I am taking the input review sentence and then constructing its vector representation (changing words into their respective index id's in embedding matrix). This vector representation will be input to the RNN.
# In order to get the word vectors, I have used Tensorflow's embedding lookup function.
mapped_reviews = []
for review in df['review']:
    temp_map = []
    for word in review.split():
        try:
            temp_map.append(word_list.index(word))
        except ValueError:
            temp_map.append(400000)
        if len(temp_map) > avg_seq_len:
            break
    # pad or truncate review w.r.t avg sequence length
    seq = np.zeros((avg_seq_len),dtype = int)
    review_arr = np.array(temp_map)
    seq[-len(temp_map):] = review_arr[-avg_seq_len:]
    mapped_reviews.append(seq)

# Mapping target labels to its numeric value for training
df.sentiment = [ 1 if each == "positive" else 0 for each in df.sentiment]

# Splitting the dataset into 80% training and 20% testing
# Not shuffling the dataset as the reviews in the csv file are added in a random manner only, thus shuffling make not much difference
x_train, x_test, y_train, y_test = mapped_reviews[:40000], mapped_reviews[40000:], df.loc[:39999,"sentiment"].values, df.loc[40000:,'sentiment'].values

# hyper-parameters for LSTM-RNN model
lstm_size = 128
lstm_layers = 1
batch_size = 128
l_rate = 0.001
n_epoch = 40

# Creating LSTM based RNN model using tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.reset_default_graph()
tf.set_random_seed(123)

# input placeholders
with tf.name_scope("inputs"):
    tf_x = tf.placeholder(dtype=tf.int32, shape=(None,avg_seq_len) , name="tf_x")
    tf_y = tf.placeholder(dtype=tf.float32, shape=(None) , name="tf_y")
    tf_keepprob = tf.placeholder(tf.float32, name="tf_keepprob")

# Call to the tf.nn.embedding_lookup() function in order to get our word vectors from the input placeholder
# The call to that function will return a 3-D Tensor of dimensionality batch size by max sequence length by word vector dimensions.
# we use lookup function to loop up the row in embedding matrix for each element in the placeholder tf_x
with tf.name_scope("embeddings"):
    embed_x = tf.nn.embedding_lookup(embeddings_vector, tf_x, name="embed_x")

# Function to create a single layer of LSTM
def lstm():
    lstm = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(lstm_size)
    return tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob= tf_keepprob)

# Can create multiple stacked layers based on lstm_layers input parameter
# Stacking these cells helps the model retain more long term dependence information, but also introduces more parameters, more training time and also model might overfit(depend on number of input samples)
with tf.name_scope("rnn_layers"):
   cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstm() for _ in range(lstm_layers)])

# tf.nn.dynamic_rnn() performs unrolling the whole network and creating a pathway for the data to flow through the RNN graph
with tf.name_scope("forward"):
    output_lstm, final_state = tf.nn.dynamic_rnn(cell, embed_x,dtype=tf.float32)


with tf.name_scope("output"):
    # giving the last output by -1
    logit = tf.layers.dense(inputs = output_lstm[:,-1],units=1,activation=None,name='logits')
    logits = tf.squeeze(logit,name="logits_squeeze")
    y_prob = tf.nn.sigmoid(logit,name='probabilities')

# defining optimizer to train the model parameters
with tf.name_scope("optimize"):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_y,logits=logits,name='cost'))
    optimizer = tf.train.AdamOptimizer(l_rate).minimize(cost,name='train_op')

# Function to create batches of data
def batch_generator(x,y,batch_size):  
    for i in range(0,len(x),batch_size):
        if y is not None:
            yield x[i:i+batch_size], y[i:i+batch_size]
        else:
            yield x[i:i+batch_size]

# We will be saving the model also for deployment and backup for long training times
saver = tf.train.Saver()

# Training LSTM_RNN model
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
        # Average training loss
        losses = losses / tot
        print("Epoch: {} , Train_loss: {}".format(epoch,losses))
        # After each 10 epochs save the model for backup
        if (epoch+1) % 10 == 0 :
            saver.save(sess,'model/sentiment-{}.ckpt'.format(epoch))

# Testing and prediction on testing data
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
accuracy = np.sum(round_ == y_test[:len(round_)])/len(round_)

# for calculating the model metrics: specificity, sensitivity and balanced accuracy
def confusion_matrix(y_pred, y_test, pos_label):
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    for (pred, true) in zip(y_pred, y_test):
        if pred == pos_label and true == pos_label:
            true_positive += 1
        elif pred == pos_label:
            false_positive += 1
        elif true == pos_label:
            false_negative += 1
        else:
            true_negative += 1
    sensi = true_positive / (true_positive + false_negative)
    spec = true_negative / (true_negative + false_positive)
    return sensi, spec

sensi, spec = confusion_matrix(round_, y_test, 1.)

# output the model matrics found after training
print("Specificity of model is-",spec)
print("Sensitivity of model is-",sensi)
print("Balanced accuracy of model is-",(spec+sensi)/2)

print("Accuracy on 10000 testing data is- ",accuracy)