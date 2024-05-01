import numpy as np
import pandas as pd

# train.csv and test.csv cannot be uploaded to github because of size. Go to https://www.kaggle.com/c/fake-news/overview , login with google or whatever
# and download the dataset. Either put this script in the same folder, or change the read_csv path.


# GUIDE
"""
id: unique id for a news article
title: the title of a news article
author: author of the news article
text: the text of the article; could be incomplete
label: a label that marks the article as potentially unreliable
    1: unreliable
    0: reliable

"""
# So we can figure out which or if we want to run on all features, or if some show bias etc.

# Owen- as for my predictions, I think Title's will have a lot of influence, because fake news obviously wants to have that clickbait. I don't really know
# why the dataset has author, maybe that will influence it in an interesting way...


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def main():

    df = pd.read_csv("train.csv")
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    # --------------------------------------------------
    # Separate titles and article bodies based on label
    reliable_df = df[df["label"] == 0]
    unreliable_df = df[df["label"] == 1]

    # Concatenate titles and article bodies for reliable and unreliable articles
    reliable_titles = " ".join(reliable_df["title"].astype(str))
    unreliable_titles = " ".join(unreliable_df["title"].astype(str))

    reliable_bodies = " ".join(reliable_df["text"].astype(str))
    unreliable_bodies = " ".join(unreliable_df["text"].astype(str))

    # Create word clouds for reliable and unreliable titles
    reliable_title_wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate(reliable_titles)
    unreliable_title_wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate(unreliable_titles)

    # Create word clouds for reliable and unreliable article bodies
    reliable_body_wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate(reliable_bodies)
    unreliable_body_wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate(unreliable_bodies)

    # Plot word clouds
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(reliable_title_wordcloud, interpolation="bilinear")
    plt.title("Word Cloud for Reliable Titles")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(unreliable_title_wordcloud, interpolation="bilinear")
    plt.title("Word Cloud for Unreliable Titles")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(reliable_body_wordcloud, interpolation="bilinear")
    plt.title("Word Cloud for Reliable Article Bodies")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(unreliable_body_wordcloud, interpolation="bilinear")
    plt.title("Word Cloud for Unreliable Article Bodies")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    # ----------------------------------------------------
    """
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

    """


main()
