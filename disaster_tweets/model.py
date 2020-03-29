""" In this competition, you’re challenged to build a machine learning model
that predicts which Tweets are about real disasters and which one’s aren’t.
https://www.kaggle.com/c/nlp-getting-started/overview """

""" Features:
1/ get exclamations count
2/ fill keywords with hashtags
3/ fill location with countries mentioned in text
4/ remove top common words for both targets
"""

import pandas as pd
import pickle
import numpy as np
import re
import nltk

# import keras
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, classification_report

from xgboost import XGBClassifier


# --------------------- Load data ---------------------
# get countries, cities from https://datahub.io/core/world-cities
# world_cities = pd.read_csv('world-cities.csv', index_col=False)
# countries = set([loc.lower() for loc in world_cities['country']])

# NLTK helpers
# nltk.download('stopwords')
# nltk.download('wordnet')
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))


# --------------------- Helper functions ---------------------
def clean_text(text):
    text = text.lower()  # text to lowercase
    text = re.sub(r"[^a-z0-9']+", " ", text)  # keep only letters and digits
    text = ' '.join(word for word in text.split() if word not in stopwords) # delete stopwors from text
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())  # lemmatize words
    text = ' '.join(stemmer.stem(word) for word in text.split())  # stem words
    return text


def get_exclamations(text):
    """ Count exclamation marks attached to words or standalone"""
    ex_count = re.findall("[a-z#]*[!]+", text.lower())
    return len(ex_count)


def get_questions(text):
    """ Count question marks attached to words or standalone"""
    quest_count = re.findall("[a-z#]*[\?]+", text.lower())
    return len(quest_count)


def get_caps(text):
    """ Count CAPS LOCK expressions """
    caps_count = re.findall("[A-Z][A-Z][A-Z][A-Z]+", text)
    return len(caps_count)


def get_hashtags(text):
    """ Extract keywords as hasshtags from tweets """
    hashtags = re.findall('#[a-z]+', text.lower())
    hashtags = ' '.join([tag.strip('#') for tag in hashtags])
    return clean_text(hashtags)


def count_targets(dataY):
    true = np.count_nonzero(dataY)
    false = len(dataY) - true
    print('=>Balance T/F: {}/{}'.format(true, false))
    return true, false


def balance_data(data):
    true, false = count_targets(data['target'])
    remove_n = abs(true - false)
    drop_id = []
    if true > false:
        drop_id = np.random.choice(data[data['target'] == 1].index, remove_n, replace=False)
    elif true < false:
        drop_id = np.random.choice(data[data['target'] == 0].index, remove_n, replace=False)
    data_balanced = data.drop(drop_id)
    true, false = count_targets(data_balanced['target'])
    return data_balanced


# ---------------------- Feature settings ------------------------
def prepare_text(data):
    # data['ex_count'] = data['text'].apply(get_exclamations)  # exclamations count
    # data['quest_count'] = data['text'].apply(get_questions)  # questions count
    # data['caps_count'] = data['text'].apply(get_caps)  # CAPS LOCK count
    # data['tags'] = data['text'].apply(get_hashtags)  # retrieve hashtags

    data['keyword'] = [word.replace('%20', ' ') if not isinstance(word, float) else np.nan for word in data['keyword']]  # clean keywords
    data['text'] = data['text'].apply(clean_text)  # clean text
    # data = data.dropna(axis=0, subset=['keyword'])  # drop missing keywords
    data['text'] = data['text'] + ' ' + data['keyword'].fillna('')  # merge keywords with text

    # select features
    # select_labels = ['target', 'ex_count', 'quest_count', 'caps_count', 'text']
    # data = data[select_labels]  #.values.reshape(-1, data[select_labels].shape[1])
    # dataY = data['target']
    # print(dataX)
    # print(dataX.shape, dataY.shape)
    return data['text']


def set_vectorizer(dataX):
    # vectorize text
    vectorizer = TfidfVectorizer(ngram_range = (1, 2),
    strip_accents = 'unicode',
    decode_error = 'replace',
    analyzer = 'word',
    min_df = 2)
    vectorized_vocabulary = vectorizer.fit(dataX)
    pickle.dump(vectorized_vocabulary, open('vectorized_vocabulary.pickle', 'wb'))

    dataX = vectorized_vocabulary.transform(dataX)
    return dataX


# ------------------------ Model settings ------------------------
def train_xgboost(trainX, trainY):
    # vectorize text features
    trainX = set_vectorizer(trainX)
    trainX = trainX.reshape(-1, trainX.shape[1])
    print('=>Vocab vector train', trainX.shape)

    selector = SelectKBest(f_classif, k=min(5000, trainX.shape[1]))
    selector.fit(trainX, trainY)
    pickle.dump(selector, open('selector.pickle', 'wb'))
    trainX = selector.transform(trainX)  #.astype('float32')
    print('=>Reduced vector train', trainX.shape)

    # booster: gbtree, gblinear, dart.
    model = XGBClassifier(max_depth=2, learning_rate=0.01, booster='gblinear', tree_method='exact', reg_alpha=0.0001, reg_lambda=0.0001, verbosity=1)
    model.fit(trainX, trainY, verbose=True)
    pickle.dump(model, open('model.pickle', 'wb'))


def predict(dataX):
    # vectorize text features
    vectorized_vocabulary = pickle.load(open('vectorized_vocabulary.pickle', 'rb'))
    dataX = vectorized_vocabulary.transform(dataX)
    print('=>Vocab vector predict', dataX.shape)

    selector = pickle.load(open('selector.pickle', 'rb'))
    dataX = selector.transform(dataX)  #.astype('float32')
    print('=>Reduced vector predict', dataX.shape)

    # load model
    model = pickle.load(open('model.pickle', 'rb'))
    # predict
    predY = model.predict(dataX)
    return predY


def run_test(testX, testY):
    predY = predict(testX)
    # model metrics
    confMatrix = confusion_matrix(testY, predY)
    report = classification_report(testY, predY)
    print(confMatrix)
    print(report)


# --------------------- Submission settings ---------------------
def get_submission(dataX):
    predY = predict(dataX)
    # write to csv
    result = pd.DataFrame({'target': predY}, index=dataX.index)
    result.to_csv('submission.csv', index=True)
    print('Submission saved.')


def main():
    data_train = pd.read_csv('train.csv', index_col=0)
    data_test = pd.read_csv('test.csv', index_col=0)

    data_train = balance_data(data_train)  # balance data

    dataX = prepare_text(data_train)
    dataY = data_train['target']

    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.05)

    train_xgboost(trainX, trainY)
    run_test(testX, testY)

    # predict test
    # dataX = prepare_text(data_test)
    # get_submission(dataX)


if __name__ == '__main__':
    main()
