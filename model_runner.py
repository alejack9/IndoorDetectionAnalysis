import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from models_config import models


# get the input attribute value associated to the best models
def get_rank1_info(result, attribute):
    return result.loc[result['rank_test_score'] == 1][attribute].values[0]


def retrieve_best_models(X_train, y_train):
    best_models = {}
    for est_name, est, params in models:
        # cross-validate: since cv=10, the validation set is 10% of the whole train dataset
        result, current_pipeline = run_crossvalidation(X_train, y_train, est, params, cv=10)
        best_models[est_name] = {'pipeline': current_pipeline}

        # retrieve metadata of the best model
        attributes = ["mean_train_score", "mean_test_score", "mean_fit_time", "mean_score_time"]
        for attribute in attributes:
            best_models[est_name][attribute] = get_rank1_info(result, attribute)

    return best_models


# cross-validation function
def run_crossvalidation(X_trainval, y_trainval, clf, params, cv=10, verbose=True):
    # params["scaler"] = [StandardScaler(), MinMaxScaler()]

    # "StandardScaler()" is a placeholder that will be change by "GridSearchCV" when "params" will be passed
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

    grid_search = GridSearchCV(pipeline, params, cv=cv, verbose=10 if verbose else 0, n_jobs=8,
                               return_train_score=True)
    grid_search.fit(X_trainval, y_trainval)

    return pd.DataFrame(grid_search.cv_results_), grid_search.best_estimator_
