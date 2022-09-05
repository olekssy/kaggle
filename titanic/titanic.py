""" Titanic binary classificcation. """

from typing import Optional

import numpy as np
import pandas as pd


class DataPrep:
    """ Class for pre-processing data for training and testing. """
    NotImplemented


class Features:
    """ Class for extracting features. """

    def __init__(self) -> None:
        raise NotImplementedError

    def fit(data: pd.DataFrame) -> None:
        raise NotImplementedError



class Model:
    """ Binary classification model. """

    def __init__(self) -> None:
        pass

    def fit(X: np.ndarray, Y: np.ndarray) -> Optional[any]:
        raise NotImplementedError

    def predict(X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


if __name__ == '__main__':
    # load data
    data_train: pd.DataFrame = pd.read_csv('data/titanic-train.csv.gz')
    data_test: pd.DataFrame = pd.read_csv('data/titanic-test.csv.gz')
