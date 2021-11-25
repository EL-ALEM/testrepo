
ELALEM_database = pd.read_csv('file.csv')
ELALEM_database.head()
# Install Libraries
!pip install textblob
!pip install tweepy
!pip install pycountry
!pip install langdetect
!pip install schedule
!pip install time
!pip install WordCloud
import schedule
import time
def job():
# Import Libraries
    from textblob import TextBlob
    import sys
    import tweepy
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    import nltk
    import pycountry
    import re
    import string

    from wordcloud import WordCloud, STOPWORDS
    from PIL import Image
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from langdetect import detect
    from nltk.stem import SnowballStemmer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from sklearn.feature_extraction.text import CountVectorizer
    nltk.download('vader_lexicon')
    # Authentication
    consumer_key = 'P1E0uYyvsjfa1V4h7syWwsLEY'
    consumer_secret = 'ATRwxB1c7WwS08MLiobdeHifhgd1mSxkZerRwAefpZLCA74aM1'
    access_token = '1336343634187350016-aXZZUw3prKzwdfG9RNC7ld2YZrTFn4'
    access_secret = 'XzHjPwtDXSdNrQutwrocy81QCJlh0Df1Hm4qLa3vp47yP'


    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    #Sentiment Analysis

    '''
    here, we are creating an algorithme that takes a hachtag ('str') and a number (int) wich will define the number of tweets to extract.
    In this part, after collecting data we will use SentimentIntensityAnalyzer() from nltk library to detect polarity score. That means: for every tweet, SentimentIntensityAnalyzer() will accord a score for 
    positivity and negativity regarding its content. Basing on this score, tweet will be considered positive (positive score > negative score), negative (negative score > positive score)
    or neutral (positive score = negative score).
    We will gather all sentences contained in same category (positive, negative or neutral) and we will transfor it in data frame format before cleaning it from all undesired values (ponctuation, duplacated tweets...) 
    in order to do same visualization graph.
    '''

    def percentage(part,whole):
        return 100 * float(part)/float(whole) 

        keyword = input("Please enter keyword or hashtag to search: ")
        noOfTweet = int(input ("Please enter how many tweets to analyze: "))


        tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en").items(noOfTweet)
        positive  = 0
        negative = 0
        neutral = 0
        polarity = 0
        neutral_list = []
        negative_list = []
        positive_list = []
        location = []
        for tweet in tweets:
            tweet_list.append(tweet.text)
            analysis = TextBlob(tweet.text)
        score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        polarity += analysis.sentiment.polarity
    
        if neg > pos:
            negative_list.append(tweet.text)
            negative += 1

        elif pos > neg:
            positive_list.append(tweet.text)
            positive += 1
    
        elif pos == neg:
            neutral_list.append(tweet.text)
            neutral += 1
        positive = percentage(positive, noOfTweet)
        negative = percentage(negative, noOfTweet)
        neutral = percentage(neutral, noOfTweet)
        polarity = percentage(polarity, noOfTweet)
        positive = format(positive, '.1f')
        negative = format(negative, '.1f')
        neutral = format(neutral, '.1f')
    tweet_list.drop_duplicates(inplace = True)
    #Cleaning Text (RT, Punctuation etc)
    
    #Creating new dataframe and new features
    tw_list = pd.DataFrame(tweet_list)
    tw_list["text"] = tw_list[0]
    
    #Removing RT, Punctuation etc
    remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
    rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
    tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
    tw_list["text"] = tw_list.text.str.lower()
    tw_list.head(10)
    #Calculating Negative, Positive, Neutral and Compound values
    
    tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
    for index, row in tw_list['text'].iteritems():
        score = SentimentIntensityAnalyzer().polarity_scores(row)
        neg = score['neg']
        neu = score['neu']
        pos = score['pos']
        comp = score['compound']
        if neg > pos:
            tw_list.loc[index, 'sentiment'] = "negative"
        elif pos > neg:
            tw_list.loc[index, 'sentiment'] = "positive"
        else:
            tw_list.loc[index, 'sentiment'] = "neutral"
        tw_list.loc[index, 'neg'] = neg
        tw_list.loc[index, 'neu'] = neu
        tw_list.loc[index, 'pos'] = pos
        tw_list.loc[index, 'compound'] = comp
    #Creating new data frames for all sentiments (positive, negative and neutral)

    tw_list_negative = tw_list[tw_list["sentiment"]=="negative"]
    tw_list_positive = tw_list[tw_list["sentiment"]=="positive"]
    tw_list_neutral = tw_list[tw_list["sentiment"]=="neutral"]
    #Function for count_values_in single columns

    def count_values_in_column(data,feature):
        total=data.loc[:,feature].value_counts(dropna=False)
        percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
        return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])
    #Count_values for sentiment
    count = count_values_in_column(tw_list,"sentiment")
    count
    
    liste_dataframes = [ELALEM_database,tw_list]
    ELALEM_database = pd.concat(liste_dataframes, sort= False)
    ELALEM_database.head()
    ELALEM_database.to_csv('file.csv')
    return 
    
schedule.every().day.at('21:09').do(job)
while True:
    schedule.run_pending()
    time.sleep(1)
    

