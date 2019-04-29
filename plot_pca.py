import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np

city = 'San Francisco'
dataPath = 'TweetScraper\\TweetScraper\\Data\\'
cityWeatherPath = dataPath + city + '_weather.csv'

cityData = pd.read_csv(cityWeatherPath, index_col=None, lineterminator='\n', dtype={'ID': str, 'TEXT': str, 'DATETIME': str, 'SENTIMENT': str,
    'WEATHER_DESCRIPTION': str, 'TEMPERATURE': np.float64, 'HUMIDITY': np.float64, 'PRESSURE': np.float64, 'WIND_SPEED': np.float64,
    'WIND_DIRECTION': np.float64})

cityData.dropna(inplace=True)

#PCA - remove irrelevant features
X = cityData[['TEMPERATURE', 'HUMIDITY', 'PRESSURE']]#, 'WIND_SPEED', 'WIND_DIRECTION' 
y = cityData['SENTIMENT']

#X = X.reset_index()
x_std = StandardScaler().fit_transform(X)
pca = decomposition.PCA(n_components=2)
sklearn_pca_x = pca.fit_transform(x_std)

sklearn_result = pd.DataFrame(sklearn_pca_x, columns=['PC1', 'PC2'])
final_result = pd.concat([sklearn_result, y], axis=1)

fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('2 component PCA - ' + city, fontsize = 20)
sentiments = ['0', '1']
colors = ['#EF553B', '#0D76BF']
for sent, color in zip(sentiments,colors):
    indicesToKeep = final_result['SENTIMENT'] == sent
    ax.scatter(final_result.loc[indicesToKeep, 'PC1']
               , final_result.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 20)
ax.legend(sentiments)
ax.grid()

print(pca.explained_variance_ratio_)
plt.show()