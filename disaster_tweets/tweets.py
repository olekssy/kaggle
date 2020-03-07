""" In this competition, you’re challenged to build a machine learning model
that predicts which Tweets are about real disasters and which one’s aren’t.
https://www.kaggle.com/c/nlp-getting-started/overview """


import pandas as pd
import pickle
import numpy as np
import re
import nltk

import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report

lemmatizer = nltk.stem.WordNetLemmatizer()

# --------------------- Load data ---------------------
data_train = pd.read_csv('train.csv', index_col=0)
data_test = pd.read_csv('test.csv', index_col=0)

# get countries, cities from https://datahub.io/core/world-cities
world_cities = pd.read_csv('world-cities.csv', index_col=False)
countries = set([loc.lower() for loc in world_cities['country']])


# --------------------- Features settings ---------------------
def clean_text(text):
    text = text.lower()  # text to lowercase
    text = re.sub(r"[^a-z']+", " ", text)  # keep only letters and digits
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())  # lemmatize tweets
    return text


def get_hashtags(text):
    """ Extract keywords as hasshtags from tweets """
    hashtags = re.findall('#[a-z0-9]+', text)
    hashtags = ' '.join([tag.strip('#') for tag in hashtags])
    return hashtags


def count_targets(dataY):
    true = np.count_nonzero(dataY)
    false = len(dataY) - true
    print('True/False: {}/{}'.format(true, false))
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


def get_features(data):
    data = balance_data(data)  # balance target values
    dataX = data['text']
    dataY = data['target']

    dataX = data['text'].apply(clean_text)  # clean text
    # data = data.dropna(how='any')
    return dataX, dataY


def set_vectorizer(dataX):
    # Text vectorizer
    n_gram = (1, 2) # n-gram sizes for tokenizing text(unigram + bigram + trigram....)
    token_mode = 'word' # whether text should be split into word or character n-grams
    min_frequency = 2 # minimum n_gram frequency below which a token will be discarded.

    vectorizer = TfidfVectorizer(ngram_range = n_gram,
                                strip_accents = 'unicode',
                                decode_error = 'replace',
                                analyzer = token_mode,
                                min_df = min_frequency)

    vectorizer_vocabulary = vectorizer.fit(dataX) # Learn vocabulary from training texts

    # Save vocabulary for further predictions usage
    pickle.dump(vectorizer_vocabulary, open("vectorizer_vocabulary.pickle", "wb"))

    return vectorizer_vocabulary


# --------------------- Model Settings ---------------------
def get_estimator(n_features):

    learning_rate = 0.01 # Initial LR (later autotuned with ReduceLROnPlateau)

    inputs = keras.Input(shape=(n_features,))
    hidden1 = keras.layers.Dense(2048, activation='relu',
                                       activity_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001))(inputs)
    hidden2 = keras.layers.LeakyReLU(alpha=0.001)(hidden1)
    hidden3 = keras.layers.Dropout(0.005)(hidden2)
    hidden4 = keras.layers.Dense(1024, activation='relu',
                                       activity_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001))(hidden3)
    hidden5 = keras.layers.LeakyReLU(alpha=0.001)(hidden4)
    hidden6 = keras.layers.Dropout(0.005)(hidden5)
    hidden7 = keras.layers.Dense(512, activation='relu',
                                      activity_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0001))(hidden6)
    hidden8 = keras.layers.LeakyReLU(alpha=0.001)(hidden7)
    hidden9 = keras.layers.Dropout(0.005)(hidden8)
    outputs = keras.layers.Dense(1, activation='sigmoid')(hidden9)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Configuration
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                    loss=keras.losses.binary_crossentropy,
                    metrics=['accuracy'])

    return model


def run_training(trainX, trainY, epochs=1):

    print('Start initial training')

    # Clean text
    # print('Start text cleaning')
    # trainX = trainX.apply(clean_text)
    # Set vectorizer
    print('Start vectorizing')
    vectorizer = set_vectorizer(trainX)
    # Vectorize text
    trainX = vectorizer.transform(trainX)

    # TUNEME: k parameter
    # Reduce vocabulary to k best predictors
    print('Start SelectKbest')
    selector = SelectKBest(f_classif, k=min(15000, trainX.shape[1]))
    selector.fit(trainX, trainY)
    pickle.dump(selector, open("selector.pickle", "wb"))
    trainX = selector.transform(trainX).astype('float32')

    # Declare model
    model = get_estimator(trainX.shape[1])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.1)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                                mode='min',
                                                patience=10,
                                                verbose=1,
                                                factor=0.1,
                                                min_lr=0.00001)

    # Train model and save
    model.fit(trainX, trainY, batch_size=20,
                              validation_split=0.2,
                              epochs=epochs,
                              callbacks=[learning_rate_reduction, es],
                              verbose=1)
    pickle.dump(model, open("model.pickle", "wb"))


def run_test(testX, testY):
    vectorizer = pickle.load(open("vectorizer_vocabulary.pickle", 'rb'))
    selector = pickle.load(open("selector.pickle", 'rb'))
    model = pickle.load(open("model.pickle", 'rb'))

    # testX = testX.apply(clean_text)
    testX = vectorizer.transform(testX)# Vectorize input
    testX = selector.transform(testX).astype('float32')# Select input

    prediction = model.predict(testX)

    # TUNEME: postprocess threshold manipulation (default 0.5)
    # prediction = np.argmax(prediction, axis=1)
    prediction = np.array([1 if i>0.5 else 0 for i in prediction.reshape(-1,)])

    print(classification_report(testY.values.reshape(-1,), prediction))


# --------------------- Submission functions ---------------------
def predict(testX):
    vectorizer = pickle.load(open("vectorizer_vocabulary.pickle", 'rb'))
    selector = pickle.load(open("selector.pickle", 'rb'))
    model = pickle.load(open("model.pickle", 'rb'))

    testX = testX.apply(clean_text)
    testX = vectorizer.transform(testX)# Vectorize input
    testX = selector.transform(testX).astype('float32')# Select input

    prediction = model.predict(testX)

    # TUNEME: postprocess threshold manipulation (default 0.5)
    # prediction = np.argmax(prediction, axis=1)
    prediction = np.array([1 if i > 0.5 else 0 for i in prediction.reshape(-1,)])
    return prediction


def get_submission(data):
    testX = data['text']
    predY = predict(testX)
    # write to csv
    result = pd.DataFrame({'id': data.index, 'target': predY})
    result = result.set_index('id')
    result.to_csv('submission.csv', index=True)
    print('Submission saved.')


dataX, dataY = get_features(data_train)
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.2, random_state=np.random.randint(0, 100), stratify=dataY)

run_training(trainX, trainY)
run_test(testX, testY)
# get_submission(data_test)
