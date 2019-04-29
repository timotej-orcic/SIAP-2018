# SIAP-2018

Dependant python libraries:
  pandas
  numpy
  dateutil
  plotly
  matplotlib
  scikit-learn
  re
  nltk
  many_stop_words
  random
  
Dependant Github projects:
  https://github.com/jonbakerfish/TweetScraper
  
-The filepath variables are set acording to the TweetScraper project because it was the root project used for fetching the tweet data, slightly modified from the original project with our download restrictions

Short description of scripts:
  lexical_classification -> creates a list of positive and negative tweets from the entire dataset (based on emoticons)
  lexical_preprocessing -> creates the positive and negative dictionaries based on the above clasification (lemmatized and stemmed)
  lexical_sentiment -> calculates the sentiment of all tweets in the given city using the above mentioned dictionaries
  
  ml_sentiment -> test script for evaluating the machine learning methodologies and their precision on our dataset
  create_test_set -> concatenates the data for the NB sentiment test set
  nb_sentiment_full -> calculates the sentiment for the remaining of the dataset (50% that was not calculated lexicaly) using NB
  
  city_weather_preprocessing -> used for concatenating the tweet data (that has sentiment) and weather data, and ploting histograms that     represent the infulence of all of the weather features on the tweet sentiment
  plot_pca -> used for plotting the 2D PCA projection of the influence of weather features on the tweet sentiment for the given city
  
Datasets and results are available here:
  https://drive.google.com/drive/folders/1HT_rerxE3aKErm-P4yqZ_g3OhI8GnpLC
