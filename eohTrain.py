import numpy as np
import pandas as pd

#train.csv and test.csv cannot be uploaded to github because of size. Go to https://www.kaggle.com/c/fake-news/overview , login with google or whatever
# and download the dataset. Either put this script in the same folder, or change the read_csv path.



#GUIDE
'''
id: unique id for a news article
title: the title of a news article
author: author of the news article
text: the text of the article; could be incomplete
label: a label that marks the article as potentially unreliable
    1: unreliable
    0: reliable

'''
# So we can figure out which or if we want to run on all features, or if some show bias etc.

# Owen- as for my predictions, I think Title's will have a lot of influence, because fake news obviously wants to have that clickbait. I don't really know
# why the dataset has author, maybe that will influence it in an interesting way...


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from wordcloud import WordCloud


def main():

    df = pd.read_csv('train.csv')
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    print(df.describe())
    print(df.head(5))


    #to start just run on title and fake/real
    X = df['title'] 
    y = df['label']

    #Actually what Im realizing if we want to only use two axis, we could potentially combine title and author or etc into one string.

    porter_stemmer = PorterStemmer() 
    
    words = stopwords.words("english")

    #I also want to use wordclouds to show what are the most common words. We could do this for title.
    # I think this could be a great part of the presentation/project
    # SO we could start by splitting into two variables depending on whether they are real or fake, and show the most common words in the real titles, and then another plot
    # for the words in the fake title.

    



main()