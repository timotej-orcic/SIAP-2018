import os
import pandas as pd
import re
import csv
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes
from sklearn.metrics import accuracy_score

from many_stop_words import get_stop_words
from nltk.corpus import stopwords

stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english'))   #About 150 stopwords
stop_words.extend(nltk_words)

dataPath = 'TweetScraper\\TweetScraper\\Data\\'
trainingPath = dataPath + 'Training_data.csv'
testPath = dataPath + 'Test_data.csv'

#UTILS
class CSVWrapper:
    def __init__(self, id, sentimentScore):
        self.id = id
        self.sentimentScore = sentimentScore

    def __iter__(self):
        return iter([self.id, self.sentimentScore])

def clean(tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        tweet = tweet.lower() 
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def removeStopWords(tweet):    
    filtered_words = []
    for w in tweet:
        if w not in stop_words:
            filtered_words.append(w)

    return filtered_words

def stem(tweet):
    ps = PorterStemmer()

    stemmed_words=[]
    for w in tweet:
        stemmed_words.append(ps.stem(w))
    
    return ' '.join(stemmed_words)
#END UTILS

#READ DATA
training_data = pd.read_csv(trainingPath)
test_data = pd.read_csv(testPath)
#END READ DATA

# Remove blank rows if any.
training_data['TEXT'].dropna(inplace=True)
test_data['TEXT'].dropna(inplace=True)

# Tokenization and filtration of stop words
training_data['TEXT'] = [word_tokenize(clean(entry)) for entry in training_data['TEXT']]
test_data['TEXT'] = [word_tokenize(clean(entry)) for entry in test_data['TEXT']]

# Remove stop words
training_data['TEXT'] = [removeStopWords(entry) for entry in training_data['TEXT']]
test_data['TEXT'] = [removeStopWords(entry) for entry in test_data['TEXT']]

# Stemming
training_data['TEXT'] = [stem(entry) for entry in training_data['TEXT']]
test_data['TEXT'] = [stem(entry) for entry in test_data['TEXT']]

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(training_data['SENTIMENT'])
Test_Y = Encoder.fit_transform(test_data['SENTIMENT'])

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(training_data['TEXT'])

Train_X_Tfidf = Tfidf_vect.transform(training_data['TEXT'])
Test_X_Tfidf = Tfidf_vect.transform(test_data['TEXT'])

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)

# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
test_data['SENTIMENT'] = predictions_NB

# Use accuracy_score function to get the accuracy
#print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y)*100)

test_data.to_csv(dataPath + 'NB_Results.csv', sep = ',', index=False)