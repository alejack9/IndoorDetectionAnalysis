from sklearn.impute import SimpleImputer

import visualization
import pandas as pd


def missing_values(X, prefix, sensor):
    X_suff = X[[col for col in X if col.startswith(prefix)]]
    # plot the percentage of missing values for each feature
    missing_values_series = pd.Series([x / len(X_suff) * 100 for x in X_suff.isna().sum()],
                                      index=[x.split('.')[-1] for x in X_suff.columns])

    visualization.plot_features_info(missing_values_series, sensor + " - Features missing values")


def features_distribution(X, prefix, sensor):
    X_suff = X[[col for col in X if col.startswith(prefix)]]
    # plot the percentage of missing values for each feature
    visualization.plot_density_all(X_suff, sensor)


def priori_analysis(X, y):
    families = X.groupby('family')
    for i, d in enumerate([families.get_group(x) for x in families.groups]):
        d = d.dropna(axis=1, how='all')
        missing_values(d, 'signals.wifi', list(families.groups.keys())[i].capitalize() + ' - WIFI')
        missing_values(d, 'signals.bluetooth', list(families.groups.keys())[i].capitalize() + ' - BLUETOOTH')
        missing_values(d, 'gps.', list(families.groups.keys())[i].capitalize() + ' - GPS')

        features_distribution(d.drop(['family', 'device', 'timestamp', 'location'], axis=1), 'signals.wifi',
                              list(families.groups.keys())[i].capitalize() + ' - WIFI')
        features_distribution(d.drop(['family', 'device', 'timestamp', 'location'], axis=1), 'signals.bluetooth',
                              list(families.groups.keys())[i].capitalize() + ' - BLUETOOTH')
        features_distribution(d.drop(['family', 'device', 'timestamp', 'location'], axis=1), 'gps.',
                          list(families.groups.keys())[i].capitalize() + ' - GPS')

    visualization.plot_class_distribution(y)
    visualization.plot_all()


def create_datasets(X):
    return [X.drop(['location', 'family', 'timestamp', 'device'], axis=1)]


def remove_nan(X_train, X_test):
    imputer = SimpleImputer(strategy="median").fit(X_train)
    X_train = imputer.transform(X_train)
    if X_test is not None:
        X_test = imputer.transform(X_test)

    return X_train, X_test
