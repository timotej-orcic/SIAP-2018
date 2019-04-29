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
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from many_stop_words import get_stop_words
from nltk.corpus import stopwords

stop_words = list(get_stop_words('en'))         #About 900 stopwords
nltk_words = list(stopwords.words('english'))   #About 150 stopwords
stop_words.extend(nltk_words)

city = 'Kansas City'
dataPath = 'TweetScraper\\TweetScraper\\Data\\'
lemPath = dataPath + city + '_lemmatized.csv'
stemPath = dataPath + city + '_stemmed.csv'
testPath = dataPath + city + '_test.csv'
tweetsPath = dataPath + city + '\\'

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

    filtered_words = [x for x in filtered_words if not hasNumbers(x)]

    return filtered_words

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

    return ' '.join(lemmatized_words)

def stem(tweet):
    ps = PorterStemmer()

    stemmed_words=[]
    for w in tweet:
        stemmed_words.append(ps.stem(w))
    
    return ' '.join(stemmed_words)

def saveToCSV(Encoder, corpus, predictions, pred_type):
    csvRows = []
    idx = 0
    for row in corpus.iterrows():
        csvItem = CSVWrapper(row[1]['ID'], predictions[idx])
        csvRows.append(csvItem)
        idx = idx + 1

    with open(dataPath + city + '_' + pred_type +  '.csv', 'w+', newline='\n', encoding='utf-8') as csv_file_stem:
        wr = csv.writer(csv_file_stem, delimiter=',')
        wr.writerow(['ID', 'SENTIMENT'])
        for tweet in csvRows:
            wr.writerow(list(tweet))

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)
#END UTILS

#READ DATA
lem_data = pd.read_csv(lemPath)
stem_data = pd.read_csv(stemPath)

print(lem_data.SENTIMENT.value_counts())
print(stem_data.SENTIMENT.value_counts())

lem_train, lem_test = train_test_split(lem_data, test_size = 0.3)
stem_train, stem_test = train_test_split(stem_data, test_size = 0.3)
#END READ DATA

#ML SENTIMENT

# Remove blank rows if any.
lem_train['TEXT'].dropna(inplace=True)
lem_test['TEXT'].dropna(inplace=True)
stem_train['TEXT'].dropna(inplace=True)
stem_test['TEXT'].dropna(inplace=True)

# Tokenization and filtration of stop words
lem_train['TEXT'] = [word_tokenize(clean(entry)) for entry in lem_train['TEXT']]
lem_test['TEXT'] = [word_tokenize(clean(entry)) for entry in lem_test['TEXT']]
stem_train['TEXT'] = [word_tokenize(clean(entry)) for entry in stem_train['TEXT']]
stem_test['TEXT'] = [word_tokenize(clean(entry)) for entry in stem_test['TEXT']]

# Remove stop words
lem_train['TEXT'] = [removeStopWords(entry) for entry in lem_train['TEXT']]
lem_test['TEXT'] = [removeStopWords(entry) for entry in lem_test['TEXT']]
stem_train['TEXT'] = [removeStopWords(entry) for entry in stem_train['TEXT']]
stem_test['TEXT'] = [removeStopWords(entry) for entry in stem_test['TEXT']]

# Stemming and lemmatization
lem_train['TEXT'] = [posTagAndLemmatize(entry) for entry in lem_train['TEXT']]
lem_test['TEXT'] = [posTagAndLemmatize(entry) for entry in lem_test['TEXT']]
stem_train['TEXT'] = [stem(entry) for entry in stem_train['TEXT']]
stem_test['TEXT'] = [stem(entry) for entry in stem_test['TEXT']]

Encoder = LabelEncoder()
Train_Y_lem = Encoder.fit_transform(lem_train['SENTIMENT'])
Test_Y_lem = Encoder.fit_transform(lem_test['SENTIMENT'])
Train_Y_stem = Encoder.fit_transform(stem_train['SENTIMENT'])
Test_Y_stem = Encoder.fit_transform(stem_test['SENTIMENT'])

Tfidf_vect_lem = TfidfVectorizer(max_features=5000)
Tfidf_vect_lem.fit(lem_train['TEXT'])
Tfidf_vect_stem = TfidfVectorizer(max_features=5000)
Tfidf_vect_stem.fit(stem_train['TEXT'])

Train_X_Tfidf_lem = Tfidf_vect_lem.transform(lem_train['TEXT'])
Test_X_Tfidf_lem = Tfidf_vect_lem.transform(lem_test['TEXT'])
Train_X_Tfidf_stem = Tfidf_vect_stem.transform(stem_train['TEXT'])
Test_X_Tfidf_stem = Tfidf_vect_stem.transform(stem_test['TEXT'])

# fit the training dataset on the NB classifier
Naive_lem = naive_bayes.MultinomialNB()
Naive_lem.fit(Train_X_Tfidf_lem, Train_Y_lem)
Naive_stem = naive_bayes.MultinomialNB()
Naive_stem.fit(Train_X_Tfidf_stem, Train_Y_stem)

# predict the labels on validation dataset
predictions_NB_lem = Naive_lem.predict(Test_X_Tfidf_lem)
predictions_NB_stem = Naive_stem.predict(Test_X_Tfidf_stem)

# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score for Lemmatized data -> ", accuracy_score(predictions_NB_lem, Test_Y_lem)*100)
print("Naive Bayes Accuracy Score for Stemed data -> ", accuracy_score(predictions_NB_stem, Test_Y_stem)*100)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM_lem = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM_lem.fit(Train_X_Tfidf_lem, Train_Y_lem)
SVM_stem = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM_stem.fit(Train_X_Tfidf_stem, Train_Y_stem)

# predict the labels on validation dataset
predictions_SVM_lem = SVM_lem.predict(Test_X_Tfidf_lem)
predictions_SVM_stem = SVM_stem.predict(Test_X_Tfidf_stem)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score for Lemmatized data -> ",accuracy_score(predictions_SVM_lem, Test_Y_lem)*100)
print("SVM Accuracy Score for Stemmed data -> ",accuracy_score(predictions_SVM_stem, Test_Y_stem)*100)

# Save to CSV files
labels_nb_lem = Encoder.inverse_transform(predictions_NB_lem)
labels_svm_lem = Encoder.inverse_transform(predictions_SVM_lem)
labels_nb_stem = Encoder.inverse_transform(predictions_NB_stem)
labels_svm_stem = Encoder.inverse_transform(predictions_SVM_stem)

saveToCSV(Encoder, lem_test, labels_nb_lem, 'nb_lem')
saveToCSV(Encoder, lem_test, labels_svm_lem, 'svm_lem')
saveToCSV(Encoder, stem_test, labels_nb_stem, 'nb_stem')
saveToCSV(Encoder, stem_test, labels_svm_stem, 'svm_stem')