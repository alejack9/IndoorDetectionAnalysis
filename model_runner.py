from joblib import dump, load
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from os import path

from models_config import models


def get_rank1_info(result, attribute):
    '''Gets the input attribute value associated to the best models'''
    return result.loc[result['rank_test_score'] == 1][attribute].values[0]


def retrieve_best_models(X_train, y_train, fs, use_saved_if_available, save_models, models_dir):
    best_models = {}
    for est_name, est, params in models:
        est_name = est_name + fs
        filename = est_name + '.joblib'

        # load the model if saved before
        if use_saved_if_available and path.exists(path.join(models_dir, filename)):
            print(f"Saved model found: {est_name}")
            # store model class in best model's pipeline
            best_models[est_name] = {'pipeline': load(path.join(models_dir, filename))}
            # load csv in result
            result = pd.read_csv(path.join(models_dir, "csvs", est_name + ".csv"))
        else:
            # cross-validate: since cv=10, the validation set is 10% of the whole train dataset
            # (80% of the whole dataset)
            result, current_pipeline = run_crossvalidation(X_train, y_train, est, params, cv=10)
            best_models[est_name] = {'pipeline': current_pipeline}
            if save_models:
                # save model's binary
                dump(best_models[est_name]['pipeline'], path.join(models_dir, filename))
                # save csv
                result.to_csv(path.join(models_dir, "csvs", est_name + ".csv"))

        # retrieve "attributes" of the best model in cross validation
        attributes = ["mean_train_score", "mean_test_score", "mean_fit_time", "mean_score_time"]
        for attribute in attributes:
            best_models[est_name][attribute] = get_rank1_info(result, attribute)
        best_models[est_name]['_all_results'] = result
    # return for each best model result and pipeline
    return best_models


def run_crossvalidation(X_trainval, y_trainval, clf, params, cv=10, verbose=True):
    # "StandardScaler()" and "RandomOverSampler" are placeholders that will be change by "GridSearchCV" when
    # "params" will be passed
    pipeline = make_pipeline(
        RandomOverSampler(random_state=42, sampling_strategy='minority'),
        StandardScaler(),
        clf)
    # GridSearchCV approach for fitting models
    grid_search = GridSearchCV(pipeline, params, cv=cv, verbose=10 if verbose else 0, n_jobs=-1,
                               return_train_score=True)
    grid_search.fit(X_trainval, y_trainval)

    return pd.DataFrame(grid_search.cv_results_), grid_search.best_estimator_
