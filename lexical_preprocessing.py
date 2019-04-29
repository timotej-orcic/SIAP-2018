import json
import nltk
import re

#used for scaling the dictionnaries
import random

from many_stop_words import get_stop_words

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#DOWNLOAD REQUIRED NLTK PACKAGES
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

dataPath = 'TweetScraper\\TweetScraper\\Data\\'

stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english'))   #About 150 stopwords
stop_words.extend(nltk_words)

#FUNCTIONS
def clean_tweet(tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        tweet = tweet.lower() 
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_wordnet_pos(treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

def tokenizeAndFilter(tweets):
    tokenized_words = []
    for text in tweets:
        tokenized_words += word_tokenize(clean_tweet(text))
    
    filtered_words = []
    for w in tokenized_words:
        if w not in stop_words:
            filtered_words.append(w)

    return filtered_words

def posTagAndLemmatize(tweets):
    pos_tagged_words = nltk.pos_tag(tweets)

    lemmatized_words = []
    lem = WordNetLemmatizer()
    for ptw in pos_tagged_words:
        lem_word = lem.lemmatize(ptw[0], get_wordnet_pos(ptw[1]))
        if lem_word not in lemmatized_words:
            lemmatized_words.append(lem_word)

    return lemmatized_words

def stem(tweets):
    ps = PorterStemmer()

    stemmed_words=[]
    for w in tweets:
        if w not in stemmed_words:
            stemmed_words.append(ps.stem(w))
    
    return stemmed_words

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
#END FUNCTIONS

negPath = 'TweetScraper\\TweetScraper\\Data\\negatives'
posPath = 'TweetScraper\\TweetScraper\\Data\\positives'

#LOAD DATA
negTweets = []
posTweets = []
with open(negPath) as n:
    lines = n.readlines()
    for tweetPath in lines:
        l = len(tweetPath)
        tweetPath = tweetPath[:l-1]
        tweetPath = tweetPath.replace('/', '\\')
        with open(tweetPath) as tw:
            data = json.load(tw)
            text = data['text']
            negTweets.append(text)

with open(posPath) as p:
    lines = p.readlines()
    for tweetPath in lines:
        l = len(tweetPath)
        tweetPath = tweetPath[:l-1]
        tweetPath = tweetPath.replace('/', '\\')
        with open(tweetPath) as tw:
            data = json.load(tw)
            text = data['text']
            posTweets.append(text)
#END LOAD DATA

#BEGIN PREPROCESSING
filtered_words_neg = tokenizeAndFilter(negTweets)
filtered_words_pos = tokenizeAndFilter(posTweets)

filtered_words_neg = [x for x in filtered_words_neg if not hasNumbers(x)]
filtered_words_pos = [x for x in filtered_words_pos if not hasNumbers(x)]

#SCALING DICTIONARIES
#posTweets = random.choices(posTweets, k=len(negTweets))
#END SCALING DICTIONARIES

#LEMMATIZATION
lemmatized_words_neg = posTagAndLemmatize(filtered_words_neg)
joint_words_neg = ' '.join(lemmatized_words_neg)
with open(dataPath + 'neg_dictionary_lemmatized', 'a+') as nd:
    nd.write(joint_words_neg)

lemmatized_words_pos = posTagAndLemmatize(filtered_words_pos)
joint_words_pos = ' '.join(lemmatized_words_pos)
with open(dataPath + 'pos_dictionary_lemmatized', 'a+') as pd:
    pd.write(joint_words_pos)
#END LEMMATIZATION

#STEMMING
stemmed_words_neg = stem(filtered_words_neg)
joint_words_neg = ' '.join(stemmed_words_neg)
with open(dataPath + 'neg_dictionary_stemmed', 'a+') as nd:
    nd.write(joint_words_neg)

stemmed_words_pos = stem(filtered_words_pos)
joint_words_pos = ' '.join(stemmed_words_pos)
with open(dataPath + 'pos_dictionary_stemmed', 'a+') as pd:
    pd.write(joint_words_pos)
#END STEMMING

#END PREPROCESSING