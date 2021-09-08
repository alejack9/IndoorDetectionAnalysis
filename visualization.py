import numpy as np
from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelEncoder
from math import ceil


def plot_class_distribution(y):
    distribution = np.unique(y, return_counts=True)
    fig, axs = plt.subplots(nrows=1, ncols=2)
    step = 1.0 / len(distribution[0])
    colors = [hsv_to_rgb(cur, 0.9, 1) for cur in np.arange(0, 1, step)]
    axs[0].bar(x=distribution[0], height=distribution[1], color=colors)
    axs[0].set_xticklabels(distribution[0], rotation=45)
    axs[1].pie(distribution[1], labels=distribution[0], autopct='%.2f%%', colors=colors)
    fig.suptitle("Number of samples for each class")


# function to plot info associated to the sensors
def plot_features_info(series, title):
    plt.figure()
    series.plot.barh()
    plt.ylabel("MACs")
    plt.title(title)


# function to plot the distribution of each sensor feature
def plot_density_all(X, title_prefix):
    plots_per_page = 12
    print(title_prefix + ': for {} rows, using {} rows'.format(len(X.columns), ceil(len(X.columns) / 2)))
    for j in range(ceil(len(X.columns) / plots_per_page)):
        cols = X.columns[j * plots_per_page:min(j * plots_per_page + plots_per_page, len(X.columns))]
        fig, axs = plt.subplots(nrows=max(ceil(len(cols) / 2), 2), ncols=2)
        for i, col in enumerate(cols):
            # print(title_prefix + ": " + str(X[col].isna().sum()) + ' / ' + str(len(X[col])))
            sbn.kdeplot(data=X, x=col, ax=axs[int(i / 2), i % 2])
            axs[int(i / 2), i % 2].set(xlabel='', ylabel='')
        fig.suptitle(
            "{} - Distribution per sensor ({}/{})".format(title_prefix, j+1, ceil(len(X.columns) / plots_per_page)))


def plot_all():
    plt.show()
