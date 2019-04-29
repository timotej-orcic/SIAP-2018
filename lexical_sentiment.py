import os
import json
import csv
import re
import nltk

from many_stop_words import get_stop_words

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

city = 'Kansas City'
tweetsPath = 'TweetScraper\\TweetScraper\\Data\\' + city + '\\'
dirPath = 'TweetScraper\\TweetScraper\\Data\\'

stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english'))   #About 150 stopwords
stop_words.extend(nltk_words)

#LOAD DICTIONARIES
with open(dirPath + 'pos_dictionary_lemmatized') as pdl:
    pos_dict_lem = pdl.read() 
pos_words_lem = pos_dict_lem.split(' ')

with open(dirPath + 'neg_dictionary_lemmatized') as ndl:
    neg_dict_lem = ndl.read()
neg_words_lem = neg_dict_lem.split(' ')

with open(dirPath + 'pos_dictionary_stemmed') as pds:
    pos_dict_stem = pds.read() 
pos_words_stem = pos_dict_stem.split(' ')

with open(dirPath + 'neg_dictionary_stemmed') as nds:
    neg_dict_stem = nds.read()
neg_words_stem = neg_dict_stem.split(' ')
#END LOAD DICTIONARIES

#UTILS
class CSVWrapper:
    def __init__(self, id, text, datetime, sentimentScore):
        self.id = id
        self.text = text
        self.datetime = datetime
        self.sentimentScore = sentimentScore

    def __iter__(self):
        return iter([self.id, self.text, self.datetime, self.sentimentScore])

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def calculateSentiment(tweet, isStem):
    tokenized_words = []
    tokenized_words += word_tokenize(clean(tweet))
    filtered_words = []
    for w in tokenized_words:
        if w not in stop_words:
            filtered_words.append(w)

    filtered_words = [x for x in filtered_words if not hasNumbers(x)]

    preprocessed_words = []
    if isStem == True:
        preprocessed_words = stem(filtered_words)
    else:
        preprocessed_words = posTagAndLemmatize(filtered_words)

    negCnt = 0
    posCnt = 0
    for word in preprocessed_words:
        if isStem == True:
            if word in neg_words_stem:
                negCnt = negCnt + 1
            if word in pos_words_stem:
                posCnt = posCnt + 1
        else:
            if word in neg_words_lem:
                negCnt = negCnt + 1
            if word in pos_words_lem:
                posCnt = posCnt + 1

    if posCnt > negCnt:
        return 1
    else: 
        return 0

def clean(tweet):
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

def posTagAndLemmatize(tweet):
    pos_tagged_words = nltk.pos_tag(tweet)

    lemmatized_words = []
    lem = WordNetLemmatizer()
    for ptw in pos_tagged_words:
        lem_word = lem.lemmatize(ptw[0], get_wordnet_pos(ptw[1]))
        lemmatized_words.append(lem_word)

    return lemmatized_words

def stem(tweet):
    ps = PorterStemmer()

    stemmed_words=[]
    for w in tweet:
        stemmed_words.append(ps.stem(w))
    
    return stemmed_words
#END UTILS

#CALC SENTIMENT
sentiments_lem = []
sentiments_stem = []
for filename in os.listdir(tweetsPath):
    filePath = tweetsPath + filename
    with open(filePath) as f:
        data = json.load(f)   
        text = data['text']

        sentiment_lem = calculateSentiment(text, False)
        csvLine_lem = CSVWrapper(filename, data['text'], data['datetime'], sentiment_lem)
        sentiments_lem.append(csvLine_lem)

        sentiment_stem = calculateSentiment(text, True)
        csvLine_stem = CSVWrapper(filename, data['text'], data['datetime'], sentiment_stem)
        sentiments_stem.append(csvLine_stem)

with open(dirPath + city + '_lemmatized' + '.csv', 'w+', newline='\n', encoding='utf-8') as csv_file_lem:
    wr = csv.writer(csv_file_lem, delimiter=',')
    wr.writerow(['ID', 'TEXT', 'DATETIME', 'SENTIMENT'])
    for tweet in sentiments_lem:
        wr.writerow(list(tweet))

with open(dirPath + city + '_stemmed' + '.csv', 'w+', newline='\n', encoding='utf-8') as csv_file_stem:
    wr = csv.writer(csv_file_stem, delimiter=',')
    wr.writerow(['ID', 'TEXT', 'DATETIME', 'SENTIMENT'])
    for tweet in sentiments_stem:
        wr.writerow(list(tweet))
#END CALC SENTIMENT