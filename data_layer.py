import pandas as pd
import numpy as np


# function to load the dataset
def load_data():
    # not using "index_col='_id.$oid'" because it makes the dataframe messier
    data = pd.read_csv('dataset/signalfingerprints.csv')
    data['target'] = data['family'] + '.' + data['location']
    # drop unwanted columns
    data = data.drop(['_id.$oid', 'Unnamed: 0', '__v'], axis=1)
    X = data.iloc[:, :-1]
    y = data['target']
    num_classes = len(np.unique(y))

    return X, y, num_classes
