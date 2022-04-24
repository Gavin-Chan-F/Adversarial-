
# coding=utf-8
import jieba
import re
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


model_save_path = "./save_data/"

def stopwordList(file):
    stopwords = [line.strip() for line in open(file, encoding='UTF-8').readlines()]
    return stopwords

def data_process(texts, labels, stopWords, maxlen, max_features, test_size=0.3, random_state=1):
    texts_seg = []
    for text in texts:
        text = re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", text)
        word_list = [word for word in jieba.lcut(text) if word not in stopWords]
        words = ' '.join(word_list)
        texts_seg.append(words)
    le = LabelEncoder().fit(labels)
    joblib.dump(le, model_save_path + "le.model")
    y = le.transform(labels)
    y = to_categorical(y)
    tokenizer = Tokenizer(num_words=max_features, lower=True)
    tokenizer.fit_on_texts(texts_seg)
    joblib.dump(tokenizer, model_save_path + "tk.model")
    #word_index = tokenizer.word_index
    x = tokenizer.texts_to_sequences(texts_seg)
    x = pad_sequences(x, maxlen=maxlen) # padding
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test
