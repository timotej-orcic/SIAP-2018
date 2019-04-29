import os
import json
import csv

cities = ['Atlanta', 'Dallas', 'Indianapolis', 'Los Angeles', 'Minneapolis', 'Nashville', 'Phoenix', 'Portland', 'San Diego', 'Seattle']
dataPath = 'TweetScraper\\TweetScraper\\Data\\'

class CSVWrapper:
    def __init__(self, city, id, text, datetime):
        self.city = city
        self.id = id
        self.text = text
        self.datetime = datetime

    def __iter__(self):
        return iter([self.city, self.id, self.text, self.datetime])

csvItems = []
for cityName in cities:
    tweetsPath = dataPath + cityName + '\\'
    for filename in os.listdir(tweetsPath):
        filePath = tweetsPath + filename
        with open(filePath) as f:
            data = json.load(f)
            text = data['text']
            datetime = data['datetime']
            csvItem = CSVWrapper(cityName, filename, text, datetime)
            csvItems.append(csvItem)

with open(dataPath + 'Test_data' + '.csv', 'w+', newline='\n', encoding='utf-8') as csv_file:
    wr = csv.writer(csv_file, delimiter=',')
    wr.writerow(['CITY','ID', 'TEXT', 'DATETIME', 'SENTIMENT'])
    for tweet in csvItems:
        wr.writerow(list(tweet))