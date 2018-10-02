# Sentiment-analysis

This repository implements sentiment classification using IMDB dataset using machine learning and natural language processing methods. I also implemented flask web app and tried to give a deployment feel to them.

I tried to solve this problem with three techniques-
- Natural language processing for sentiment analysis using [AFINN word valence](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010). 

- Sentiment classification using [Bag Of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) .

- Sentiment analysis using multilayer [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) based [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) network.

## Natural language processing for sentiment analysis using AFINN word valence

- AFINN is a word list for sentiment analysis. It has around 2477 English words and phrases with their affinity of being a positive word or negative word represented in a scale of -5 (most negative word) to +5 (most positive word).
- In this basic technique I tried to preprocess an input review into a token of words using some preprocessing steps, then I tend to find each word in the review(processed) onto the AFINN word list and noted its proximity of being positive or being negative.
- Added up the proximity for each word and predicted the review is positive or negative under some threshold(I have chosen 0 as provided a good result on some loose checkings).
- In the training predictions, I have also used multi-threading to speed up the process also to increase the corpora of words I have used wordnet inc. the word valence dictionary size to 5496 from 2477 originally.
- It is a language processing technique which works on some hard-coded rules(like choosing the threshold etc) and tries to get the opinion of the text on the word it contains.
> On the test set of IMDB dataset, this model got an accuracy of - 76.1% 

## Sentiment classification using Bag Of Words and Machine learning

- This is a machine learning technique, in which we try to represent the corpus of words into a bag of word representation using Tfidf representation technique (applying some preprocessing to the text side by side). 
- Then I tried to fit a Multinomial naive Bayes classifier onto this Tfidf representation.
- To get the prediction I transformed the test review using the above fitted Tfidf representation and got the prediction.
>  On the test set of IMDB dataset, this model got an accuracy of - 89.3%

## Sentiment analysis using multilayer LSTM

- Above two technique work on the principle that given a word, how much that word belongs to a certain category and we apply it to whole text of words and try to get the intuition that which category does this whole text of word belong to.
- But these model fail to capture word to word relationships i.e, how much given a word affect the occurrence of its nearby words.
- Each word in a sentence depends greatly on what came before and comes after it. In order to account for this dependency, we use a recurrent neural network.
- RNN is very good at getting dynamic temporal behavior for a time sequence and we can use this behavior to get a relationship among words in a text.
- Especially LSTM (Long Short-Term Memory) units are modules that you can place inside of recurrent neural networks and they make sure that the hidden state is able to encapsulate information about long-term dependencies in the text.
> On the test set of IMDB dataset, this model got an accuracy of - 87.1%
 
 ## Sentiment analysis using CNN text classification
 
- CNN has proven to be not only good in computer vision but has provided some state of the art results in natural language processing.
- The input layer to CNN comprised of word embeddings, followed by multiple filters, then a max-pooling layer than to a sigmoid classifier with some dropout added in the last layer.
- Here I have used embeddings(low-dimensional representations) for words included in the training procedure from scratch but some research papers have proven that word2vec or glove embedding with some tweaking in multiple channels has given more good results.
- Also, I have used word level CNN as character level CNN have not provided better results in the initial checkings.
> On the test set of IMDB dataset, this model got an accuracy of - 90.04%
 
 ## How to use this tool...
 
 1. Just clone this repository
 2. From cloned directory run the app.py file in the command prompt
 3. Play with this tool ;)
 
 #### Prerequisites -
 For running - Tensorflow, flask, numpy
 
 For training - Sklearn, Pandas, NLTK
 
 > These all can be installed using pip.  
