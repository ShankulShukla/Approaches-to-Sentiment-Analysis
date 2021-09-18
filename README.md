# Approaches optimizing sentiment analysis for IMDB movie reviews

This repository implements sentiment classification using the IMDB dataset using machine learning and natural language processing methods. I also implemented a flask web app and tried to give a deployment feel to them.

I tried to solve this problem with three techniques-
- Natural language processing for sentiment analysis using [AFINN word valence](http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010). 

- Sentiment classification using [Bag Of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) .

- Sentiment analysis using multilayer [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) based [RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network) network.
## Demo 
> Movie review flask app based on various model created.

![image](https://drive.google.com/uc?export=view&id=1jEGDukR-olW28IcQzrxhjm5lnQFH_Igd)

## Lexicon based classifier using AFINN word valence

- AFINN is a word list for sentiment analysis. It has around 2477 English words and phrases with their affinity of being a positive word or negative word represented on a scale of -5 (most negative word) to +5 (most positive word).
- In this basic technique I tried to preprocess an input review into a token of words using some preprocessing steps, then I tend to find each word in the review(processed) onto the AFINN word list and noted its proximity of being positive or being negative.
- Added up the proximity for each word and predicted the review is positive or negative under some threshold(I have chosen 0 as provided a good result on some loose checkings).
- In the training predictions, I have also used multi-threading to speed up the process also to increase the corpora of words I have used wordnet inc. the word valence dictionary size to 5496 from 2477 originally.
- It is a language processing technique that works on some hard-coded rules(like choosing the threshold etc) and tries to get the opinion of the text on the word it contains.
> ``` 
> Specificity of the model (in percent)- 68.74%
> Sensitivity of the model (in percent)- 73.56%
> Balanced accuracy of model - 71.15%
> ```

## Multinomial naive bayes classifier using n-gram features of TF-IDF vectors

- This is a machine learning technique, in which we try to represent the corpus of words into a bag of word representation using the Tfidf representation technique (applying some preprocessing to the text side by side). 
- Then I tried to fit a Multinomial naive Bayes classifier onto this Tfidf representation.
- To get the prediction I transformed the test review using the above fitted Tfidf representation and got the prediction.

**Using unigram features-**
> ``` 
> Specificity of the model (in percent)- 88.704%
> Sensitivity of the model (in percent)- 84.901%
> Balanced accuracy of model - 86.8025%
> ```

**Using bigram features-**
> ``` 
> Specificity of the model (in percent)- 92.769%
> Sensitivity of the model (in percent)- 86.179%
> Balanced accuracy of model - 89.474%
> ```

## Multilayer LSTM-RNN classifier using custom-trained word embedding and pre-trained GloVe word embedding

- Above two techniques work on the principle that given a word, how much that word belongs to a certain category and we apply it to the whole text of words and try to get the intuition that which category does this whole text of word belong to.
- But these model fail to capture word to word relationships i.e, how much given a word affect the occurrence of its nearby words.
- Each word in a sentence depends greatly on what came before and comes after it. In order to account for this dependency, we use a recurrent neural network.
- RNN is very good at getting dynamic temporal behavior for a time sequence and we can use this behavior to get a relationship among words in a text.
- Especially LSTM (Long Short-Term Memory) units are modules that you can place inside of recurrent neural networks and they make sure that the hidden state can encapsulate information about long-term dependencies in the text.

**Using GloVe word embedding-**
> ``` 
> Specificity of the model (in percent)- 87.702%
> Sensitivity of the model (in percent)- 86.37%
> Balanced accuracy of model - 87.036%
> ```

**Using custom trained word embedding-**
> ``` 
> Specificity of the model (in percent)- 88.148%
> Sensitivity of the model (in percent)- 88.052%
> Balanced accuracy of model - 88.1%
> ```
 
 ## 1-D word-level CNN text classifier using custom trained word embedding
 
- CNN has proven to be not only good in computer vision but has provided some state-of-the-art results in natural language processing.
- The input layer to CNN is comprised of word embeddings, followed by multiple filters, then a max-pooling layer then to a sigmoid classifier with some dropout added in the last layer.
- Here I have used embeddings(low-dimensional representations) for words included in the training procedure from scratch but some research papers have proven that word2vec or glove embedding with some tweaking in multiple channels has given more good results.
- Also, I have used word-level CNN as character-level CNN has not provided better results in the initial checkings.

> ``` 
> Specificity of the model (in percent)- 88.98%
> Sensitivity of the model (in percent)- 91.1%
> Balanced accuracy of model - 90.04%
> ```
 
 ## Usage and training
 
 > Python version > 3.x is recommended 
 
 **Clone the repo**
 
 **First install dependencies** 
 
 ```
 pip install -r requirements.txt
 ```
 **Custom train any classifier**
 
 ```
 python <filename>.py
 ```
 
 **Deploy the movie review flask app**
 
 ```
 python app.py
 ```
 
 > All trained models and data will be available in this [link]()

## References 
- Sherstinsky A. Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) network. Physica D: Nonlinear Phenomena 2020 Mar;404:132306.
- Kim, Y. Convolutional neural networks for sentence classification. arXiv 2014, arXiv:1408.5882
- Pal, Subarno & Ghosh, Soumadip & Nag, Amitava. (2018). Sentiment Analysis in the Light of LSTM Recurrent Neural Networks. International Journal of Synthetic Emotions. 9. 33-39. 10.4018/IJSE.2018010103
- Finn Årup Nielsen, "A new ANEW: evaluation of a word list for sentiment analysis in microblogs", Proceedings of the ESWC2011 Workshop on 'Making Sense of Microposts': Big things come in small packages. Volume 718 in CEUR Workshop Proceedings: 93-98. 2011 May. Matthew Rowe, Milan Stankovic, Aba-Sah Dadzie, Mariann Hardey (editors)
- Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
