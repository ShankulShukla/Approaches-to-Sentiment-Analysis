# Sentiment-analysis

This repository implements sentiment classification using IMDB dataset using machine learning and natural language processing methods. I also implemented flask web app for each and tried to give a production feel to them.

I tried to solve this problem with three techniques-
- Natural language processing for sentiment analysis using [AFINN word valence](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010). 

- Sentiment classification using [Bag Of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) .

- Sentiment analysis using multilayer [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) based [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) network.

## Natural language processing for sentiment analysis using AFINN word valence

- AFINN is a word list for sentiment analysis. It has around 2477 english words and phrases with their affinity of being positive word or negative word represented in a scale of -5 (most negative word) to +5 (most positive word).
- In this basic technique I tried to preprocess a input review 
