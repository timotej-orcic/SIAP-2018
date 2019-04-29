# SIAP-2018

Dependant python libraries: <br />
- pandas
&nbsp;&nbsp;&nbsp; numpy <br />
&nbsp;&nbsp;&nbsp; dateutil <br />
&nbsp;&nbsp;&nbsp; plotly <br />
&nbsp;&nbsp;&nbsp; matplotlib <br />
&nbsp;&nbsp;&nbsp; scikit-learn <br />
&nbsp;&nbsp;&nbsp; re <br />
&nbsp;&nbsp;&nbsp; nltk <br />
&nbsp;&nbsp;&nbsp; many_stop_words <br />
&nbsp;&nbsp;&nbsp; random <br />
  
Dependant Github projects: <br />
&nbsp;&nbsp;&nbsp; https://github.com/jonbakerfish/TweetScraper <br />
  
-The filepath variables are set acording to the TweetScraper project because it was the root project used for fetching the tweet data, slightly modified from the original project with our download restrictions <br />

Short description of scripts: <br />
&nbsp;&nbsp;&nbsp; lexical_classification -> creates a list of positive and negative tweets from the entire dataset (based on emoticons) <br />
&nbsp;&nbsp;&nbsp; lexical_preprocessing -> creates the positive and negative dictionaries based on the above clasification (lemmatized and stemmed) <br />
&nbsp;&nbsp;&nbsp; lexical_sentiment -> calculates the sentiment of all tweets in the given city using the above mentioned dictionaries <br />
  
&nbsp;&nbsp;&nbsp; ml_sentiment -> test script for evaluating the machine learning methodologies and their precision on our dataset <br />
&nbsp;&nbsp;&nbsp; create_test_set -> concatenates the data for the NB sentiment test set <br />
&nbsp;&nbsp;&nbsp; nb_sentiment_full -> calculates the sentiment for the remaining of the dataset (50% that was not calculated lexicaly) using NB <br />
  
&nbsp;&nbsp;&nbsp; city_weather_preprocessing -> used for concatenating the tweet data (that has sentiment) and weather data, and ploting histograms that represent the infulence of all of the weather features on the tweet sentiment <br />
&nbsp;&nbsp;&nbsp; plot_pca -> used for plotting the 2D PCA projection of the influence of weather features on the tweet sentiment for the given city <br />
  
Datasets and results are available here: <br />
&nbsp;&nbsp;&nbsp; https://drive.google.com/drive/folders/1HT_rerxE3aKErm-P4yqZ_g3OhI8GnpLC <br />
