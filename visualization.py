import pandas as pd
import numpy as np
from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt
import seaborn as sbn
from math import ceil

from sklearn.metrics import plot_confusion_matrix


def save_fig(fig, title):
    fig.set_size_inches(19.2, 10.8)
    fig.savefig(
        fname="./figs/{}.png".format("".join([world.capitalize() for world in title.split(' ')])
                                     .replace('/', "Of")),
        format="png", transparent=False,
        dpi=150)


def plot_class_distribution(y, sub=''):
    distribution = np.unique(y, return_counts=True)
    print(pd.DataFrame(
        {"Family": [sub if sub != '' else 'Global'], "Std": [np.std(distribution[1])],
         "Avg": [np.mean(distribution[1])],
         "Max": [np.max(distribution[1])], "Min": [np.min(distribution[1])],
         "Rsd": [np.std(distribution[1]) / np.mean(distribution[1]) * 100]}).set_index("Family"))
    fig, axs = plt.subplots(nrows=1, ncols=2)
    step = 1.0 / len(distribution[0])
    colors = [hsv_to_rgb(cur, 0.9, 1) for cur in np.arange(0, 1, step)]
    axs[0].bar(x=distribution[0], height=distribution[1], color=colors)
    axs[0].set_xticklabels(distribution[0], rotation=45)
    axs[1].pie(distribution[1], labels=distribution[0], autopct='%.2f%%', colors=colors)
    fig.suptitle("Number of samples for each class {}".format("- " + sub if sub != "" else ""))
    save_fig(fig, "Number of samples for each class {}".format("- " + sub if sub != "" else ""))


# function to plot info associated to the sensors
def plot_features_info(series, title):
    min_value = max(series.min() - (1 - (0.01 * series.min())), 0)  # min_value is 0 when x < 0.99009901
    plt.figure()
    plt.xlim(min_value, 100)
    plt.xticks(list(plt.xticks()[0]))  # in order to force to show the first "tick"
    series.plot.barh()
    plt.ylabel('')
    plt.title(title)
    save_fig(plt.gcf(), title)


def plot_importance(series, xlabel, title):
    data = pd.Series()
    for sensor in ['gps', 'bluetooth', 'wifi']:
        sensor_importance_sum = series[[x for x in series.index if sensor in x]].sum()
        if sensor_importance_sum:
            data[sensor] = sensor_importance_sum
    if len(data.keys()) <= 1:
        print('Not enough sensors to plot, skipping...')
        return
    plt.figure()
    ax = data.sort_values(ascending=False).plot.bar(rot=0)
    plt.xlabel(xlabel)
    for p in ax.patches:
        ax.annotate(str(round(p.get_height() * 100) / 100) + "%", (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylabel('')
    plt.title(title)
    save_fig(plt.gcf(), title)


# function to plot the distribution of each sensor feature
def plot_density_all(X, title_prefix):
    plots_per_page = 12
    print(f'{title_prefix}: for {len(X.columns)} rows, using {ceil(len(X.columns) / 2)} rows')
    for j in range(ceil(len(X.columns) / plots_per_page)):
        cols = X.columns[j * plots_per_page:min(j * plots_per_page + plots_per_page, len(X.columns))]
        fig, axs = plt.subplots(nrows=max(ceil(len(cols) / 2), 2), ncols=2)
        for i, col in enumerate(cols):
            # print(title_prefix + ": " + str(X[col].isna().sum()) + ' / ' + str(len(X[col])))
            sbn.kdeplot(data=X, x=col, ax=axs[int(i / 2), i % 2], warn_singular=False, label=col)
            axs[int(i / 2), i % 2].set(xlabel='', ylabel='')
            axs[int(i / 2), i % 2].legend(labels=[col.split(".")[-1]])
        fig.suptitle(f"{title_prefix} - Distribution per sensor ({j + 1}/{ceil(len(X.columns) / plots_per_page)})")
        save_fig(fig, f"{title_prefix} - Distribution per sensor ({j + 1}/{ceil(len(X.columns) / plots_per_page)})")


def plot_all():
    return
    # plt.show()


def plot_feature_score_correlation(feature_score_correlation, title):
    fig, ax = plt.subplots()
    ax.plot(feature_score_correlation.iloc[:, 0], feature_score_correlation['mean_train_score'],
            label='Train Score', marker='o')
    ax.plot(feature_score_correlation.iloc[:, 0], feature_score_correlation['mean_test_score'],
            label='Test Score', marker='o')
    ax.set_xlabel(feature_score_correlation.columns[0].split('__')[-1])
    ax.set_ylim(0.3, 1.05)
    ax.legend()
    fig.suptitle(title)
    save_fig(fig, title)


# function to plot the confusion matrices of each model
def plot_confusion_matrices(models, X, y, n_cols=3):
    n_rows = ceil(len(models) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    for i, (name, model) in enumerate(models.items()):
        if n_rows > 1:
            ax = axs[int(i / n_cols), i % n_cols]
        else:
            ax = axs[i % n_cols]
        plot_confusion_matrix(model['pipeline'], X, y, ax=ax, xticks_rotation=45)
        ax.set_title(' '.join(name.split('_')[:-1]))
    fig.suptitle(f"Confusion Matrices per Model (Features Count: {X.shape[1]})")
    save_fig(fig, f"Confusion Matrices per Model (Features Count: {X.shape[1]})")


# one plot for each set of models (grouped by 'dataset size')
def plot_accuracies(scores_table, n_cols=3):
    n_rows = ceil(len(scores_table) / n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    for i, accuracies_table in enumerate(scores_table):
        # sort on validation score
        accuracies_table = accuracies_table.sort_values(by=['mean_test_score'], ascending=False, axis=1)
        X_axis = np.arange(len(accuracies_table.columns))
        if n_rows > 1:
            ax = axs[int(i / n_cols), i % n_cols]
        else:
            ax = axs[i % n_cols]

        # two bars: one for train score and one for validation score
        ax.bar(X_axis - 0.2, accuracies_table.loc['mean_train_score'], 0.4, label='Train Score')
        ax.bar(X_axis + 0.2, accuracies_table.loc['mean_test_score'], 0.4, label='Val Score')

        # show percentages on top
        for p in ax.patches:
            ax.annotate(str(round(p.get_height() * 100 * 100) / 100) + "%",
                        (p.get_x() * 1.005, p.get_height() * 1.005))
        ax.legend(loc='lower right')
        plt.sca(ax)
        plt.ylim(0, 1.1)
        plt.xticks(X_axis, [' '.join(x.split('_')[:-1]) for x in accuracies_table.columns], rotation=30)
        ax.set_ylabel("Score")
        ax.set_title(f"Features Count: {accuracies_table.columns[0].split('_')[-1]}")
    fig.suptitle('Validation accuracies per Dataset')
    save_fig(fig, 'Validation accuracies per Dataset')


# function that takes a series with the models as input and returns a dataframe with models grouped by dataset size
def group_models(series, models_names, subsets_sizes):
    data = [[series[model_name + fs] for fs in subsets_sizes] for model_name in models_names]
    col = [s[1:] + " features" for s in subsets_sizes]
    return pd.DataFrame(data, columns=col, index=models_names)


# function to display one only plot with testing scores
def plot_testing_accuracy(scores_table, models_names, subsets_sizes):
    df = group_models(scores_table, models_names, subsets_sizes)
    ax = df.plot.bar(rot=0)
    plt.xticks(range(len(models_names)), [x.replace('_', ' ') for x in models_names])
    for p in ax.patches:
        ax.annotate(str(round(p.get_height() * 100 * 100) / 100) + "%",
                    (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.title('Testing accuracies per Dataset')
    save_fig(plt.gcf(), 'Testing accuracies per Dataset')


def plot_losses(losses):
    plt.figure()
    for fs, loss in losses.items():
        plt.plot(loss, label=f"{fs.replace('_', '')} features")
    plt.yscale('log')
    plt.ylabel('Loss value (log scale)')
    plt.xlabel('Epoch')
    plt.title("Loss Progression for Dataset")
    plt.legend()
    save_fig(plt.gcf(), "Loss Progression for Dataset")
