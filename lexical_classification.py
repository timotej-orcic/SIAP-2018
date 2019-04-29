import json
import os
from os import listdir

dirPath = 'TweetScraper\\TweetScraper\\Data\\'

smileyTweets = []
sadTweets = []
for city in os.listdir(dirPath):
    if city != 'WEATHER DATA':
        for filename in os.listdir(dirPath + city):
            filePath = dirPath + city + '\\' + filename
            with open(filePath) as f:
                data = json.load(f)
                text = data['text']
                if (':)' in text) or (':-)' in text) or ('=)' in text) or (':D' in text):
                    smileyTweets.append(filePath + '\n')
                elif (':(' in text) or (':-(' in text) or ('=(' in text) or (';(' in text):
                    sadTweets.append(filePath + '\n')

with open(dirPath + 'positives', 'a+') as p:
    for tweetPath in smileyTweets:
        p.write(tweetPath)

with open(dirPath + 'negatives', 'a+') as n:
    for tweetPath in sadTweets:
        n.write(tweetPath)