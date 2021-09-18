# Approaches to optimize sentiment analysis for IMDB movie reviews

In this research-based repository, I tried to tackle the sentiment classification of movie reviews using optimized machine learning, deep learning, and natural language processing methods. 

## Demo 
> Movie review flask app based on the various models created.

![image](https://drive.google.com/uc?export=view&id=1jEGDukR-olW28IcQzrxhjm5lnQFH_Igd)

## Lexicon based classifier using AFINN word valence

- AFINN is a word list for sentiment analysis. It has around 2477 English words and phrases with their affinity of being a positive word or negative word represented on a scale of -5 (most negative word) to +5 (most positive word). As AFINN has limited word mapping, so to add extended corpora of words for the task, I used NLTK synset (wordnet) instances to a group of synonymous words and add that to our lexicon mapping.
- To do any text analysis, it is very important to note the relation between different parts of a sentence, to achieve this I inferred English language structure to define the following rules -
  - Degree modifiers to alter sentiment intensity. *Positive intensifier*: Ex- "very" when used "good", intensify its sentiment i.e., "very good" has more positive sentiment then "good", *Negative intensifier*: Ex- "very" when used "bad", intensify its sentiment i.e., "very bad" has more negative sentiment then "bad".
  - Negative contractions and negations are used to negate ideas. For example - "good" has a positive connotation to it but "not good" has a strong negative connotation, so a "not" negations changes idea of being good.
  - Identifying words are all in capital letters, people do use word-shape to signal emphasis, example - "movie was BAD" have more negative intensity than "movie was bad".
  - Identifying exclamations in the review as people conventionally do use punctuation to signal increased sentiment intensity. In this implementation considering exclamation symbol, for example - "great!!!" have more positive intensity than "great".
  - If a word is not found in word list mapping, check for its stem or root word.
- In this model I preprocess an input review into a token of words using preprocessing steps, then I tend to find each word in the review(processed) onto the AFINN word list and noted its proximity of being positive or being negative.
- Fine-tune Exclamationfactor, Capsfactor, Negationfactor, Positiveintensifactor, Negativeintensifactor to optimize meaning transfer among parts of sentences. Refer code for explanation. 

> ``` 
> Specificity of the model (in percent)- 68.74%
> Sensitivity of the model (in percent)- 73.56%
> Balanced accuracy of model - 71.15%
> ```

## Multinomial naive Bayes classifier using n-gram features of TF-IDF vectors

- This is a machine learning technique, in which we try to represent the corpus of words into a bag of word representation using the Tfidf representation technique (applying some preprocessing to the text side by side). 
- Using TF-IDF to tokenize documents, learn the vocabulary, and inverse document frequency weightings as vectorized classifier input. Extensive English language analysis to develop pre-processing for the classifier *(similar to lexicon model, refer code for a clear view)*.
- Extracting emoticons and appending at the end of the review, as emoticons in my analysis carry special meaning so not removing them.
- Using the n-gram approach, for example for text "great movie", bigram "great movie" make a lot more sense than unigrams "great" and "movie". In this implementation, I have used unigram and bigram features. 
- As stop words generally do not pertain to any special meaning, not considering in this implementation and removing stop word from review.
- In this model, I prepend the prefix "not" to every word after a token of logical negation(negation list above) until the next punctuation mark. EX- "I did not like this movie", in this example classifier will interpret "like" as a different feature(positive feature) ignoring its association with a negation "not". After adding the negation prefix, "like" becomes "notlike" will thus occur more often in a negative document and act as cues for negative sentiment. Similarly, words like "notbad" will acquire positive associations with positive sentiment
- sklearn's multinomial naive Bayes algorithm as a classifier to fit the sentiment model using processed features as input. Using partial fit only to allow incremental learning.

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
- Although I tried to overcome this, but these models are not that much confident in tackling long-term and complex word to word relationships.
- RNN is very good at getting dynamic temporal behavior for a time sequence and we can use this behavior to get a relationship among words in a text.
- In this implementation, we are using RNN with Long Short Term Memory Units. These units make sure that the hidden state vector is able to maintain long-term dependencies in the text.
-  I am using word vector representation as input to RNN, I have to transform text information to these vectors as words similar in context, meaning, and semantics reside's in relatively the same area in the vector space since they both have similar definitions and are both used in similar contexts.
- In this implementation, I am using a pre-trained word vector i.e., GloVe it much a much more manageable matrix than word2vec due to its size. The matrix will contain 400,000-word vectors, each with a dimensionality of 50, and also I will be training my own word vector representation from the corpora of words available in the IMDB dataset.
- Refer to code to understand the review pre-processing and RNN architecture.

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
 
 ## 1-D word-level CNN text classifier using a custom trained word embedding
 
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
