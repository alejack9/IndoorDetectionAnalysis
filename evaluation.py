from string import digits

import visualization
import pandas as pd


def add_test_scores(current_bests, X_test, y_test):
    '''Adds the testing scores to the models metadata'''
    for name, info in current_bests.items():
        current_bests[name]['final_test_score'] = current_bests[name]['pipeline'].score(X_test, y_test)
    return current_bests


def partial_results_analysis(models, X_test, y_test, X_cols):
    '''Plots confusion matrices and feature importance (retrieved by RandomForest) of passed models'''
    # plot confusion matrices of each model
    visualization.plot_confusion_matrices(models, X_test, y_test, n_cols=2)

    # display the importance of each feature based on Random Forest model
    rf = models["RandomForest_" + str(len(X_cols))]["pipeline"].named_steps.randomforestclassifier
    rank_var = pd.Series(rf.feature_importances_ * 100, index=X_cols).sort_values(ascending=False)
    visualization.plot_importance(rank_var, xlabel='Importance Score (%)',
                                  title=f"Features Importance (Features Count: {len(X_cols)})")
    visualization.plot_all()


def results_analysis(best_models, subsets_sizes: list[str], losses):
    '''Plots validation and test accuracies of each model; also plots losses if passed'''
    pd_models = pd.DataFrame(best_models)
    # replace every digits with none, remove the last character (_) and take them without repeats
    models_names: list[str] = pd.unique(
        [name.translate({ord(k): None for k in digits})[:-1] for name in pd_models.columns])

    # plot validation scores grouped by dataset
    scores_table_per_dataset = [pd_models[[col for col in pd_models if col.endswith(fs)]] for fs in subsets_sizes]
    visualization.plot_accuracies(scores_table_per_dataset, n_cols=2)

    # plot testing score of each model
    visualization.plot_testing_accuracy(pd_models.transpose()['final_test_score'], models_names, subsets_sizes)

    # plot losses if passed
    if len(losses.keys()) > 0:
        visualization.plot_losses(losses)
    visualization.plot_all()
