# SIAP-2018

Dependant python libraries: <br />
  pandas <br />
  numpy <br />
  dateutil <br />
  plotly <br />
  matplotlib <br />
  scikit-learn <br />
  re <br />
  nltk <br />
  many_stop_words <br />
  random <br />
  
Dependant Github projects: <br />
  https://github.com/jonbakerfish/TweetScraper <br />
  
-The filepath variables are set acording to the TweetScraper project because it was the root project used for fetching the tweet data, slightly modified from the original project with our download restrictions <br />

Short description of scripts: <br />
  lexical_classification -> creates a list of positive and negative tweets from the entire dataset (based on emoticons) <br />
  lexical_preprocessing -> creates the positive and negative dictionaries based on the above clasification (lemmatized and stemmed) <br />
  lexical_sentiment -> calculates the sentiment of all tweets in the given city using the above mentioned dictionaries <br />
  
  ml_sentiment -> test script for evaluating the machine learning methodologies and their precision on our dataset <br />
  create_test_set -> concatenates the data for the NB sentiment test set <br />
  nb_sentiment_full -> calculates the sentiment for the remaining of the dataset (50% that was not calculated lexicaly) using NB <br />
  
  city_weather_preprocessing -> used for concatenating the tweet data (that has sentiment) and weather data, and ploting histograms that        represent the infulence of all of the weather features on the tweet sentiment <br />
  plot_pca -> used for plotting the 2D PCA projection of the influence of weather features on the tweet sentiment for the given city <br />
  
Datasets and results are available here: <br />
  https://drive.google.com/drive/folders/1HT_rerxE3aKErm-P4yqZ_g3OhI8GnpLC <br />
