import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

models = [
    (
        "k-NN",
        KNeighborsClassifier(),
        {
            'kneighborsclassifier__n_neighbors': np.concatenate([[3, 6, 16, 36, 84], [2, 4, 10, 24, 55, 128]])
            # np.logspace(1, 7, 11, base=2, dtype=np.int)
        }
    ),
    (
        'DecisionTree',
        DecisionTreeClassifier(random_state=42),
        {
        }
    ),
    (
        'RandomForest',
        RandomForestClassifier(random_state=42, n_jobs=4),
        {
            # 'clf__criterion': ['gini', 'entropy'],  # since gini works well, we don't need to check entropy
            'randomforestclassifier__n_estimators': np.concatenate([[599, 664, 734, 813, 899], [10, 26, 70, 188,
                                                                                                499]])
            # np.logspace(2.7781512503836434, 2.9542425094393248, 5, base=10, dtype=np.int)]
        }
    )
]

for _, _, params in models:
    params['randomoversampler'] = [None, RandomOverSampler(random_state=42, sampling_strategy='minority')]
    params['standardscaler'] = [StandardScaler(), MinMaxScaler()]
