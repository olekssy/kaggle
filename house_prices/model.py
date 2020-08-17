""" With 79 explanatory variables describing (almost) every aspect of residential
homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/ """


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
import scipy
import xgboost

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate


# -------------------- Assumptions --------------------
# Feature labels
labelTarget = 'SalePrice'
labelCathegory = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'MiscFeature', 'GarageQual', 'GarageType', 'GarageCond', 'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'MasVnrType', 'OverallCond', 'YrSold', 'MoSold')
labelNumerical = ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea')
labelReal = ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType')


# -------------------- Helper functions --------------------
def bootstrap_nans(data, labelCathegory=labelCathegory, labelNumerical=labelNumerical, labelReal=labelReal):
    """ Interpolate missing data """
    labelAll = labelCathegory + labelNumerical + labelReal
    for k in labelAll:
        if k in labelCathegory:
            data[k] = data[k].fillna('None')
        elif k in labelNumerical:
            data[k] = data[k].fillna(0)
        elif k in labelReal:
            data[k] = data[k].fillna(data[k].mode()[0])
    # treat special cases
    data = data.drop(['Utilities'], axis=1)
    data["Functional"] = data["Functional"].fillna("Typ")
    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
    return data


def boxcox_transform(data, maxSkew=0.75, _lambda=0.15):
    """ Transform skewed features via Box Cox """
    num_keys = data.dtypes[data.dtypes != 'object'].index
    skewed_features = data[num_keys].apply(lambda x: scipy.stats.skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_features})
    skewness = skewness[abs(skewness) > maxSkew]
    skewed_features = skewness.index
    for feat in skewed_features:
        data[feat] = scipy.special.boxcox1p(data[feat], _lambda)
    return data


def transform_data(data):
    """ Transform numerical features into cathegorical, add meta-features """
    # cathegorical data transform
    data['MSSubClass'] = data['MSSubClass'].apply(str)
    data['OverallCond'] = data['OverallCond'].astype(str)
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)
    # add meta-feature
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    return data


def normalize(data):
    """ Normalize Y distribution, remove anomalies """
    try:
        # lognormalize labelTarget variable
        data[labelTarget] = np.log1p(data[labelTarget])
        # remove outliers
        data = data.drop(data[(data['GrLivArea']>4000) & (data[labelTarget]<300000)].index)
    except KeyError:
        pass
    return data


def cathegory_encoder(data, labelCathegory=labelCathegory):
    """ Encode cathegorical labels """
    for k in labelCathegory:
        encoder = sklearn.preprocessing.LabelEncoder()
        encoder.fit(list(data[k].values))
        data[k] = encoder.transform(list(data[k].values))
    return data


# -------------------- Feature settings --------------------
def prepare_data(data):
    """ Feature engineering pipeline """
    data = bootstrap_nans(data)  # fill missing values
    data = transform_data(data)  # transform select numerical data into cathegorical
    data = cathegory_encoder(data)  # encode cathegorical features
    data = boxcox_transform(data)  # box cox transform skewed features
    data = pd.get_dummies(data)  # convert cathegorical features
    return data


# -------------------- Training settings --------------------
def train_model(dataX, dataY):
    """ Train XGBoost model """
    model = xgboost.XGBRegressor(
        objective='reg:squarederror',
        booter='gblinear',
        tree_method='exact',
        colsample_bytree=0.5,
        subsample=0.5,
        learning_rate=0.05,
        max_depth=3,
        reg_alpha=0.05,
        reg_gamma=0.05,
        reg_lambda=0.05,
        n_estimators=2000)
    model.fit(dataX.values.reshape(-1, dataX.shape[1]), dataY)
    pickle.dump(model, open('model.pickle', 'wb'))


# -------------------- Testing settings --------------------
def predict(dataX):
    """ Predict dependent variable from a features vector """
    # load model
    model = pickle.load(open('model.pickle', 'rb'))
    # predict
    predY = model.predict(dataX.values.reshape(-1, dataX.shape[1]))
    return predY


def test_model(dataX, dataY):
    """ Test model on test sample """
    predY = predict(dataX)

    # regression metrics
    MSE = sklearn.metrics.mean_squared_error(dataY, predY)
    R2 = sklearn.metrics.r2_score(dataY, predY)
    MAE = sklearn.metrics.median_absolute_error(dataY, predY)

    print('=== Test results ===')
    print(f'R^2 (1.0 is the best) = {R2:4.4f}')
    print(f'MSE (0.0 is the best) = {MSE:4.4f}')
    print(f'MAE (0.0 is the best) = {MAE:4.4f}\n')
    return R2


def cross_validate_model(dataX, dataY, n_splits=4):
    """ K-folds cross-validation """
    # reshape features vector
    dataX = dataX.values.reshape(-1, dataX.shape[1])
    # load model, set splits, compute scores
    model = pickle.load(open('model.pickle', 'rb'))
    k_folds = KFold(n_splits=n_splits, shuffle=True)
    scores = cross_validate(model, dataX, dataY, cv=k_folds, scoring=('neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'))

    # get metrics
    MSE = scores['test_neg_mean_squared_error'].mean()
    R2 = scores['test_r2'].mean()
    MAE = scores['test_neg_mean_absolute_error'].mean()

    print('=== Cross-validation ===')
    print(f'R^2 (1.0 is the best) = {R2:4.4f}')
    print(f'MSE (0.0 is the best) = {-MSE:4.4f}')
    print(f'MAE (0.0 is the best) = {-MAE:4.4f}')
    print(f'N-folds = {n_splits}\n')


# -------------------- Submission settings --------------------
def get_submission(dataX):
    """ Submit prediction to csv for kaggle """
    predY = np.expm1(predict(dataX))
    # write to csv
    result = pd.DataFrame({labelTarget: predY}, index=dataX.index)
    # print(result)
    result.to_csv('submission.csv', index=True)
    print('Submission saved.')


def main():
    # load data
    dataTrain = pd.read_csv('train.csv', index_col='Id', header=0)
    dataTest = pd.read_csv('test.csv', index_col='Id', header=0)

    # prepare data
    dataTrain = normalize(dataTrain)  # normalize data, remove outliers
    dataAll = pd.concat((dataTrain, dataTest))
    dataAll = prepare_data(dataAll)

    # split prepared features into train/test samples
    dataTrainProc = dataAll[dataAll[labelTarget].notnull()]
    dataTestProc = dataAll[dataAll[labelTarget].isnull()].drop([labelTarget], axis=1)

    # split train/test
    dataTrainX = dataTrainProc.drop([labelTarget], axis=1)  # select X
    dataTrainY = dataTrain[labelTarget]  # select Y
    trainX, testX, trainY, testY = sklearn.model_selection.train_test_split(dataTrainX, dataTrainY, test_size=0.2)

    # train, validate, test model
    train_model(trainX, trainY)
    cross_validate_model(trainX, trainY, n_splits=4)
    test_model(testX, testY)

    # get submission
    get_submission(dataTestProc)


if __name__ == '__main__':
    main()
