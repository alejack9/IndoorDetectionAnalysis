import visualization
import pandas as pd

# return columns which start by prefix specified in parameter
def get_dataset_prefix(X, prefix):
    return X[[col for col in X if col.startswith(prefix)]]

# 
def missing_values(X, prefix, sensor):
    X_pref = get_dataset_prefix(X, prefix)
    # plot the percentage of missing values for each feature
    missing_values_series = pd.Series([x / len(X_pref) * 100 for x in X_pref.isna().sum()],
                                      index=[x.split('.')[-1] for x in X_pref.columns])

    visualization.plot_features_info(missing_values_series, sensor + " - Features missing values")


def features_distribution(X, prefix, sensor):
    # plot the percentage of missing values for each feature
    visualization.plot_density_all(get_dataset_prefix(X, prefix), sensor)


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

        visualization.plot_all()

    [visualization.plot_class_distribution(y.loc[y.str.contains(family_name)], family_name.capitalize()) for family_name
     in families.groups.keys()]
    visualization.plot_class_distribution(y)
    visualization.plot_all()


def create_datasets(X):
    base = X.drop(['location', 'family', 'timestamp', 'device'], axis=1)
    X_subsets = [base,
                 base.drop(['gps.latitude', 'gps.longitude', 'gps.altitude', 'gps.accuracy'], axis=1),
                 base[['gps.latitude', 'gps.longitude', 'gps.altitude', 'gps.accuracy']]
                 ]

    subsets_sizes = ["_" + str(len(df.columns)) for df in X_subsets]

    return X_subsets, subsets_sizes


def remove_nan(X_train, X_test):
    X_train = pd.DataFrame(X_train).fillna(-100, inplace=False).to_numpy()
    if X_test is not None:
        X_test = pd.DataFrame(X_test).fillna(-100, inplace=False).to_numpy()

    return X_train, X_test
