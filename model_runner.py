from joblib import dump, load
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from os import path

from models_config import models


# get the input attribute value associated to the best models
def get_rank1_info(result, attribute):
    return result.loc[result['rank_test_score'] == 1][attribute].values[0]


def retrieve_best_models(X_train, y_train, fs, use_saved_if_available, save_models, models_dir):
    best_models = {}
    for est_name, est, params in models:
        est_name = est_name + fs
        filename = est_name + '.joblib'

        if use_saved_if_available and path.exists(path.join(models_dir, filename)):
            # load the model
            print(f"Saved model found: {est_name}")
            best_models[est_name] = {'pipeline': load(path.join(models_dir, filename))}
            result = pd.read_csv(path.join(models_dir, "csvs", est_name + ".csv"))
        else:
            # cross-validate: since cv=10, the validation set is 10% of the whole train dataset
            result, current_pipeline = run_crossvalidation(X_train, y_train, est, params, cv=10)
            best_models[est_name] = {'pipeline': current_pipeline}
            if save_models:
                # save the model and the associated info
                dump(best_models[est_name]['pipeline'], path.join(models_dir, filename))
                result.to_csv(path.join(models_dir, "csvs", est_name + ".csv"))

        # retrieve metadata of the best model
        attributes = ["mean_train_score", "mean_test_score", "mean_fit_time", "mean_score_time"]
        for attribute in attributes:
            best_models[est_name][attribute] = get_rank1_info(result, attribute)
        best_models[est_name]['_all_results'] = result
    return best_models


# cross-validation function
def run_crossvalidation(X_trainval, y_trainval, clf, params, cv=10, verbose=True):
    # "StandardScaler()" and "RandomOverSampler" are placeholders that will be change by "GridSearchCV" when
    # "params" will be passed
    pipeline = make_pipeline(
        RandomOverSampler(random_state=42, sampling_strategy='minority'),
        StandardScaler(),
        clf)

    grid_search = GridSearchCV(pipeline, params, cv=cv, verbose=10 if verbose else 0, n_jobs=-1,
                               return_train_score=True)
    grid_search.fit(X_trainval, y_trainval)

    return pd.DataFrame(grid_search.cv_results_), grid_search.best_estimator_
