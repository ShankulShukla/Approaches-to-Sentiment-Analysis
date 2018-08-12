# Sentiment-analysis

This repository implements sentiment classification using IMDB dataset using machine learning and natural language processing methods. I also implemented flask web app for each and tried to give a deployment feel to them.

I tried to solve this problem with three techniques-
- Natural language processing for sentiment analysis using [AFINN word valence](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010). 

- Sentiment classification using [Bag Of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) .

- Sentiment analysis using multilayer [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) based [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) network.

## Natural language processing for sentiment analysis using AFINN word valence

- AFINN is a word list for sentiment analysis. It has around 2477 english words and phrases with their affinity of being positive word or negative word represented in a scale of -5 (most negative word) to +5 (most positive word).
- In this basic technique I tried to preprocess a input review into token of words using some preprocessing steps, then I tend to find each word in the review(processed) onto the AFINN word list and noted the its proximity of being positive or being negative.
- Added up the proximity for each word and predicted the review being positive or negative under some threshold(I have choosen 0 as provided good result on loose checkings). 
- It is a language processing technique which work on some hard coded rules(like choosing the threshold etc) and tries to get the opinion of the text on the word it contains.
> On test set of imdb dataset, this model got a accuracy of - 76.1% 

## Sentiment classification using Bag Of Words and Machine learning

- This is machine learning technique, in which we try to represent corpus of words into a bag of word representation using Tfidf representation technique (applying some preprocessing to text side by side). 
- Then I tried to fit a Multinomial naive bayes classifier onto this Tfidf representaion.
- To get the prediction I transformed the test review using the above fitted Tfidf representaion and got the prediction.
>  On test set of imdb dataset, this model got a accuracy of - 89.3%

## Sentiment analysis using multilayer LSTM

- Above two technique work on the principle that given a word, how much that word belong to certain category and we apply it to whole text of words and try to get the intution that which category does this whole text of word belong to.
- But these model fail to capture word to word relationships i.e, how much given a word affect the occurance of its nearby words.
- Each word in a sentence depends greatly on what came before and comes after it. In order to account for this dependency, we use a recurrent neural network.

