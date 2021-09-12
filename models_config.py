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
            'kneighborsclassifier__n_neighbors': np.logspace(1, 8, 10, base=2, dtype=np.int)
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
            'randomforestclassifier__n_estimators': [200, 300, 400, 500, 600, 700, 800, 1000]
        }
    )
]

for _, _, params in models:
    params['randomoversampler'] = [None, RandomOverSampler(random_state=42, sampling_strategy='minority')]
    params['standardscaler'] = [StandardScaler(), MinMaxScaler()]
