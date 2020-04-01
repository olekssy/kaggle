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
def xgboost_regression(dataX, dataY):
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
    pickle.dump(model, open('xgb_model.pickle', 'wb'))


# -------------------- Testing settings --------------------
def predict(dataX):
    # load model
    model = pickle.load(open('xgb_model.pickle', 'rb'))
    # predict
    predY = model.predict(dataX.values.reshape(-1, dataX.shape[1]))
    return predY


def run_test(dataX, dataY):
    # predict
    # lognormalize data and predict
    predY = predict(dataX)
    RMSE = np.sqrt(sklearn.metrics.mean_squared_error(dataY, predY))
    print('RMSE = {:4.4f}'.format(RMSE))


# -------------------- Submission settings --------------------
def get_submission(dataX):
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

    # train, test model
    xgboost_regression(trainX, trainY)
    run_test(testX, testY)

    # get submission
    get_submission(dataTestProc)


if __name__ == '__main__':
    main()
