import nltk
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.layers import LSTM, Activation, SpatialDropout1D
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import re

train_dir = "data/train.csv"
df = pd.read_csv(train_dir)

df = df.dropna()
df = df.reset_index()
X = df.drop(labels=['label', 'id'], axis=1)
y = df['label']
