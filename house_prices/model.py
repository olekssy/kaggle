""" With 79 explanatory variables describing (almost) every aspect of residential
homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/ """


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def get_features(data):
    selectFeatures = ['MSSubClass', 'LotFrontage', 'LotArea', 'MiscVal', 'MoSold', 'YrSold']
    dataX = data[selectFeatures]
    dataX = dataX.fillna(0)
    return dataX


def run_training(trainX, trainY):
    model = LinearRegression()
    model.fit(trainX.values.reshape(-1, trainX.shape[1]), trainY)
    score = model.score(trainX, trainY)
    print('R^2 train = {:4.2f}'.format(score))
    pickle.dump(model, open('model.pickle', 'wb'))


def run_test(testX, testY):
    model = pickle.load(open('model.pickle', 'rb'))
    # predict
    score = model.score(testX.values.reshape(-1, testX.shape[1]), testY)
    print('R^2 test = {:4.2f}'.format(score))


def predict(dataX):
    # load model
    model = pickle.load(open('model.pickle', 'rb'))
    # predict
    predY = model.predict(dataX.values.reshape(-1, dataX.shape[1]))
    return predY


def get_submission(dataX):
    predY = predict(dataX)
    # write to csv
    result = pd.DataFrame({'SalePrice': predY}, index=dataX.index)
    result.to_csv('submission.csv', index=True)
    print('Submission saved.')


def main():
    # load data
    dataTrain = pd.read_csv('train.csv', index_col='Id', header=0)
    dataTest = pd.read_csv('test.csv', index_col='Id', header=0)

    # prepare data, split train/test
    dataX = get_features(dataTrain)
    dataY = dataTrain['SalePrice']
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.2)

    # train model
    run_training(trainX, trainY)

    # test model
    run_test(testX, testY)

    # get submission
    dataX = get_features(dataTest)
    get_submission(dataX)


if __name__ == '__main__':
    main()
