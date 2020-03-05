""" Titanic: Machine Learning from Disaster
Predict which passengers survived the Titanic shipwreck.
https://www.kaggle.com/c/titanic/overview """

import pickle
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB

from xgboost import XGBClassifier


# ------------------------ Helper Functions ------------------------
def scale_feature(data):
    scaler = preprocessing.MinMaxScaler()
    dataScaled = scaler.fit_transform(data)
    return dataScaled

def encode_words(data):
    encoder = preprocessing.OrdinalEncoder().fit(data)
    dataEnc = encoder.transform(data)
    return dataEnc

def count_survivors(dataY):
    """ DEPRECATED """
    survived = np.count_nonzero(dataY)
    died = len(dataY) - survived
    print('Survived/Died: {}/{}'.format(survived, died))
    return survived, died


def balance_data(data):
    """ DEPRECATED """
    survived, died = count_survivors(data['Survived'])
    remove_n = abs(survived - died)
    dropId = []
    if survived > died:
        dropid = np.random.choice(data[data['Survived'] == 1].index, remove_n, replace=false)
    elif survived < died:
        dropId = np.random.choice(data[data['Survived'] == 0].index, remove_n, replace=False)
    dataBalanced = data.drop(dropId)
    return dataBalanced


# ------------------------ Features Settings ------------------------
def get_features(data):
    # scale numerical features
    data['Age'] = scale_feature(data['Age'].fillna(data['Age'].median()).values.reshape(-1, 1))
    data['Fare'] = scale_feature(data['Fare'].fillna(data['Fare'].median()).values.reshape(-1, 1))
    # encode cathegorical features
    data['Sex'] = encode_words(data['Sex'].values.reshape(-1, 1))
    data['Embarked'] = encode_words(data['Embarked'].fillna('A').values.reshape(-1, 1))
    # select features
    xlabels = ['Pclass', 'Age', 'SibSp', 'Fare', 'Parch', 'Sex', 'Embarked']
    dataX = data[xlabels].values.reshape(-1, len(xlabels))
    return dataX


# ------------------------ Model settings ------------------------
def log_regression(trainX, trainY):
    """ Logistic regression with CV """
    model = LogisticRegressionCV(penalty='l2', solver='liblinear')
    model.fit(trainX, trainY)
    pickle.dump(model, open('model.pickle', 'wb'))


def naive_bayes(trainX, trainY):
    """ The Complement Naive Bayes classifier """
    model = ComplementNB()
    model.fit(trainX, trainY)
    pickle.dump(model, open('model.pickle', 'wb'))


def svm_rbf(trainX, trainY):
    """ SVM model. Kernel: linear, poly, rbf, sigmoid """
    model = svm.SVC(kernel='rbf')
    model.fit(trainX, trainY)
    pickle.dump(model, open('model.pickle', 'wb'))


def random_forest(trainX, trainY):
    """ A random forest classifier. """
    model = RandomForestClassifier(n_estimators=20, max_depth=5)
    model.fit(trainX, trainY)
    pickle.dump(model, open('model.pickle', 'wb'))


def decision_tree(trainX, trainY):
    """ A decision tree classifier. Criterion: gini, entropy """
    model = DecisionTreeClassifier(criterion='gini', max_depth=3)
    model.fit(trainX, trainY)
    pickle.dump(model, open('model.pickle', 'wb'))


def mlp_classifier(trainX, trainY):
    model = MLPClassifier(
                        hidden_layer_sizes=(128, ),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        batch_size=20,
                        learning_rate='adaptive',
                        learning_rate_init=0.001,
                        early_stopping=True)
    model.fit(trainX, trainY)
    pickle.dump(model, open('model.pickle', 'wb'))


def xgboost_classifier(trainX, trainY):
    model = XGBClassifier(max_depth=2, learning_rate=0.001, reg_alpha=0.0001, reg_lambda=0.0001)
    model.fit(trainX, trainY)
    pickle.dump(model, open('model.pickle', 'wb'))


# ------------------------ Test settings ------------------------
def predict(dataX):
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
    f1 = f1_score(testY, predY)
    # print(confMatrix)
    # print(report)
    return f1


def train_test_all(trainX, testX, trainY, testY):
    log_regression(trainX, trainY)
    print('log_regression_f1 = {:.2f}'.format(run_test(testX, testY)))
    naive_bayes(trainX, trainY)
    print('naive_bayes_f1 = {:.2f}'.format(run_test(testX, testY)))
    svm_rbf(trainX, trainY)
    print('svm_rbf_f1 = {:.2f}'.format(run_test(testX, testY)))
    random_forest(trainX, trainY)
    print('random_forest_f1 = {:.2f}'.format(run_test(testX, testY)))
    decision_tree(trainX, trainY)
    print('decision_tree_f1 = {:.2f}'.format(run_test(testX, testY)))
    mlp_classifier(trainX, trainY)
    print('mlp_classifier_f1 = {:.2f}'.format(run_test(testX, testY)))
    xgboost_classifier(trainX, trainY)
    print('xgboost_classifier_f1 = {:.2f}'.format(run_test(testX, testY)))


def get_submission(data):
    dataX = get_features(data)
    predY = predict(dataX)
    # write to csv
    result = pd.DataFrame({'Survived': predY}, index=data.index)
    result.to_csv('submission.csv', index=True)
    print('Submission saved.')


def main():
    # load data
    dataTrain = pd.read_csv('train.csv', index_col='PassengerId', header=0)
    dataTest = pd.read_csv('test.csv', index_col='PassengerId', header=0)

    # [DEPRECATED] balance data
    # dataTrain = balance_data(dataTrain)

    # split train/test
    dataX = get_features(dataTrain)
    dataY = dataTrain['Survived'].values
    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.2)

    # train select model
    train_test_all(trainX, testX, trainY, testY)
    # xgboost_classifier(trainX, trainY)

    # test model
    # run_test(testX, testY)

    # predict test sample, save submission
    # get_submission(dataTest)
    pass


if __name__ == '__main__':
    main()
