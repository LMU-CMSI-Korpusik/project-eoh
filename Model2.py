import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
# from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# been having errors with stopwords???
print(stopwords.words("english"))

df = pd.read_csv("train.csv")
print(df.shape)
print(df.head())
df.fillna(" ", inplace=True)

# So for this strategy I wanted to combine title and content, so that we can use that. Just a random idea I had.
df["content"] = df["title"] + " " + df["author"]
print("\n combined title and body \n")
print(df.head())


port_stem = PorterStemmer()
# Stemming I learned is reducing word to base form, so like removing suffices and prefixes.


def stemming(content):
    # replace any non-alphabetic characters in the content variable with a space character
    stemmed_content = re.sub("[^a-zA-Z]", " ", content)
    # Convert all words into lower case letters
    stemmed_content = stemmed_content.lower()
    # Split the words into list
    stemmed_content = stemmed_content.split()
    # generate a list of stemmed words from stemmed_content, excluding any stop words from the list
    stemmed_content = [
        port_stem.stem(word)
        for word in stemmed_content
        if not word in stopwords.words("english")
    ]
    # Join the elements from the list 'stemmed_content' into a single string separated by spaces
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content


df["content"] = df["content"].apply(stemming)
# print(df['content'])


# now converting text data to numerical data.
transformer = TfidfTransformer(smooth_idf=False)
count_vectorizer = CountVectorizer(ngram_range=(1, 2))
counts = count_vectorizer.fit_transform(df["content"].values)
tfidf = transformer.fit_transform(counts)


print("\n splitting the data...\n")
# splitting data
targets = df["label"].values
# print(f"target shape: {targets.shape}")
# print(f"X shape: {tfidf.shape}")
X_train, X_test, y_train, y_test = train_test_split(
    tfidf, targets, test_size=0.2, random_state=49
)


# Next step
def train(model, model_name):
    model.fit(X_train, y_train)
    print(f"Training accuracy of {model_name} is {model.score(X_train,y_train)}")
    print(f"testing accuracy of {model_name} is {model.score(X_test,y_test)}")


def conf_matrix(model):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)


def class_report(model):
    print(classification_report(y_test, model.predict(X_test)))


print("\ntraining the model...\n")
model_lr = (
    LogisticRegression()
)  # Used chatgpt for some of these parts, I always forget the syntax specifics
train(model_lr, "LogisticRegression")

print(conf_matrix(model_lr))

print(class_report(model_lr))  # chatgpt help


# NOW TRYING SVM
