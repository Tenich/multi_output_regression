import numpy as np
import pandas
import pickle
import gzip
import datetime
from sklearn.preprocessing import Normalizer

to_log = ["Quan_4", "Quan_5", "Quan_6", "Quan_7", "Quan_8", "Quan_9", 
          "Quan_10", "Quan_11", "Quan_12", "Quan_13", "Quan_14", "Quan_15", 
          "Quan_16", "Quan_17", "Quan_18", "Quan_19", "Quan_21", "Quan_22", "Quan_27", 
          "Quan_28", "Quan_29", "Quant_22", "Quant_24", "Quant_25"]

def create_dataset(dataframe_train, dataframe_test):
    global to_log
    dataframe = pandas.concat([dataframe_train, dataframe_test])
    dataframe['Date_3'] = dataframe.Date_1 - dataframe.Date_2
    train_size = dataframe_train.shape[0]
    X_categorical = []
    X_quantitative = []
    X_date = []
    X_id = []
    ys = np.zeros((train_size,12), dtype=np.int)
    columns = []
    for col in dataframe.columns:
        if col.startswith('Cat_'):
            columns.append(col)
            uni = np.unique(dataframe[col])
            if len(uni) > 1:
                # Quick smart way to binarize categorical variables:
                X_categorical.append(uni==dataframe[col].values[:,None])
        elif col.startswith('Quan_') or col.startswith('Quant_'):
            columns.append(col)
            # Use logscale when needed:
            if col in to_log:
                dataframe[col] = np.log(1+dataframe[col])
            # if the column is not just full of NaN:
            if (pandas.isnull(dataframe[col])).sum() > 1:
                tmp = dataframe[col].copy()
                # median imputation:
                tmp = tmp.fillna(tmp.median())
                X_quantitative.append(tmp.values)
        elif col.startswith('Date_'):
            columns.append(col)
            # if the column is not just full of NaN:
            tmp = dataframe[col].copy()
            if (pandas.isnull(tmp)).sum() > 1:
                # median imputation:
                tmp = tmp.fillna(tmp.median())
            X_date.append(tmp.values[:,None])
            # extract day/month/year to catch seasonal effects
            year = np.zeros((tmp.size,1))
            month = np.zeros((tmp.size,1))
            day = np.zeros((tmp.size,1))
            for i, date_number in enumerate(tmp):
                date = datetime.date.fromordinal(int(date_number))
                year[i,0] = date.year
                month[i,0] = date.month
                day[i,0] = date.day
            X_date.append(year)
            X_date.append(month)
            X_date.append(day)
            # consider year, month day as categorical and create
            # binary representation:
            X_date.append((np.unique(year)==year).astype(np.int))
            X_date.append((np.unique(month)==month).astype(np.int))
            X_date.append((np.unique(day)==day).astype(np.int))
        elif col=='id':
            pass # X_id.append(dataframe[col].values)
        elif col.startswith('Outcome_'):
            outcome_col_number = int(col.split('M')[1]) - 1
            tmp = dataframe[col][:train_size].copy()
            # median imputation:
            tmp = tmp.fillna(tmp.median())
            ys[:,outcome_col_number] = tmp.values
        else:
            raise NameError

    X_categorical = np.hstack(X_categorical).astype(np.float)
    X_quantitative = np.vstack(X_quantitative).astype(np.float).T
    X_date = np.hstack(X_date).astype(np.float)

    X = np.hstack([X_categorical, X_quantitative, X_date])
    X_train = X[:train_size,:]
    X_test = X[train_size:,:]
    return X_train, X_test, ys, columns


def redundant_columns(X):
    """Identify which columns are redundant.
    """
    idx = []
    for i in range(X.shape[1]-1):
        for j in range(i+1, X.shape[1]):
            if (X[:,i] == X[:,j]).all() :
                idx.append(j)
    return np.unique(idx)

def load():
    filename_train = './data/online_sales/TrainingDataset.csv'
    filename_test = './data/online_sales/TestDataset.csv'
    dataframe_train = pandas.read_csv(filename_train)
    dataframe_test = pandas.read_csv(filename_test)

    ids = dataframe_test.values[:,0].astype(np.int)

    X_train, X_test, ys, columns = create_dataset(dataframe_train, dataframe_test)

    X = np.vstack([X_train, X_test])
    X = Normalizer().fit_transform(X)
    idx = redundant_columns(X)
    columns_to_keep = list(set(range(X.shape[1])).difference(set(idx.tolist())))
    X = X[:,columns_to_keep]
    X_train = X[:X_train.shape[0], :]
    X_test = X[X_train.shape[0]:, :]

    return {'online_sales': (X_train, ys)}