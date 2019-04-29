import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
import plotly as py

city = 'New York City'
dataPath = 'TweetScraper\\TweetScraper\\Data\\'
weatherDataPath = dataPath + 'WEATHER DATA\\'
weatherDescriptionPath = weatherDataPath + 'weather_description.csv'
temperaturePath = weatherDataPath + 'temperature.csv'
humidityPath = weatherDataPath + 'humidity.csv'
pressurePath = weatherDataPath + 'pressure.csv'
windSpeedPath = weatherDataPath + 'wind_speed.csv'
windDirPath = weatherDataPath + 'wind_direction.csv'
cityDataPath = dataPath + city + '_stemmed.csv'

def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
               +timedelta(hours=t.minute//30))

cityData = pd.read_csv(cityDataPath)
#cityData = cityData[:1000]

negCnt = cityData.loc[cityData.SENTIMENT == 0, 'SENTIMENT'].count()
posCnt = cityData.loc[cityData.SENTIMENT == 1, 'SENTIMENT'].count()
print('Negatives cnt: ' + str(negCnt))
print('Positives cnt: ' + str(posCnt))

weather_description = pd.read_csv(weatherDescriptionPath)
temperature = pd.read_csv(temperaturePath)
humidity = pd.read_csv(humidityPath)
pressure = pd.read_csv(pressurePath)
wind_speed = pd.read_csv(windSpeedPath)
wind_direction = pd.read_csv(windDirPath)

cityWeatherDesc = []
cityTemp = []
cityHum = []
cityPress = []
cityWS = []
cityWD = []
for datetime in cityData['DATETIME']:
    dt = parser.parse(datetime)
    roundDateTime = hour_rounder(dt)
    rdtStr = str(roundDateTime)    
    idxListWD = weather_description.index[weather_description['datetime'] == rdtStr].tolist()
    idxListTMP = temperature.index[temperature['datetime'] == rdtStr].tolist()
    idxListHUM = humidity.index[humidity['datetime'] == rdtStr].tolist()
    idxListPR = pressure.index[pressure['datetime'] == rdtStr].tolist()
    idxListWS = wind_speed.index[wind_speed['datetime'] == rdtStr].tolist()
    idxListWD = wind_direction.index[wind_direction['datetime'] == rdtStr].tolist()

    if not idxListWD:
        weatherDesc = 0.0
    else:
        wdIndex = idxListWD[0]
        weatherDesc = weather_description.loc[wdIndex, 'Kansas City']

    if not idxListTMP:
        temp = 0.0
    else:
        tmpIndex = idxListTMP[0]
        temp = temperature.loc[tmpIndex, 'Kansas City']

    if not idxListHUM:
        hum = 0.0
    else:
        humIndex = idxListHUM[0]
        hum = humidity.loc[humIndex, 'Kansas City']

    if not idxListPR:
        pre = 0.0
    else:
        preIndex = idxListPR[0]
        pre = pressure.loc[preIndex, 'Kansas City']

    if not idxListWS:
        ws = 0.0
    else:
        wsIndex = idxListWS[0]
        ws = wind_speed.loc[wsIndex, 'Kansas City']

    if not idxListWD:
        wd = 0.0
    else:
        wdIndex = idxListWD[0]
        wd = wind_direction.loc[humIndex, 'Kansas City']

    cityWeatherDesc.append(weatherDesc)
    cityTemp.append(temp)
    cityHum.append(hum)
    cityPress.append(pre)
    cityWS.append(ws)
    cityWD.append(wd)

cityData['WEATHER_DESCRIPTION'] = cityWeatherDesc
cityData['TEMPERATURE'] = cityTemp
cityData['HUMIDITY'] = cityHum
cityData['PRESSURE'] = cityPress
cityData['WIND_SPEED'] = cityWS
cityData['WIND_DIRECTION'] = cityWD

X = cityData[['WEATHER_DESCRIPTION', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION']].values
y = cityData['SENTIMENT'].values

# plotting histograms
data = []

legend = {0:False, 1:False, 2:False, 3:False, 4:False, 5:True}

colors = {1: '#0D76BF',
          0: '#EF553B'}

for col in range(6):
    for key in colors:
        trace = dict(
            type='histogram',
            x=list(X[y==key, col]),
            opacity=0.75,
            xaxis='x%s' %(col+1),
            marker=dict(color=colors[key]),
            name=key,
            showlegend=legend[col]
        )
        data.append(trace)

layout = dict(
    barmode='overlay',
    xaxis=dict(domain=[0, 0.15], title='weather_description'),
    xaxis2=dict(domain=[0.17, 0.32], title='temperature'),
    xaxis3=dict(domain=[0.34, 0.49], title='humidity'),
    xaxis4=dict(domain=[0.51, 0.66], title='pressure'),
    xaxis5=dict(domain=[0.68, 0.83], title='wind_speed'),
    xaxis6=dict(domain=[0.85, 1], title='wind_direction'),
    yaxis=dict(title='count'),
    title='Distribution of the different weather features'
)

fig = dict(data=data, layout=layout)
py.offline.plot(fig, filename='exploratory-vis-histogram-' + city)

cityData.to_csv(dataPath + city + '_weather.csv', sep=',', encoding='utf-8', index=False)