import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

models = [
    (
        "k-NN",
        KNeighborsClassifier(),
        {
            'clf__n_neighbors': np.logspace(1, 9, 10, base=2, dtype=np.int)
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
        RandomForestClassifier(random_state=42, n_jobs=8),
        {
            # 'clf__criterion': ['gini', 'entropy'] # since gini works well, we don't need to check entropy
            'clf__n_estimators': [10, 20, 50, 100, 200, 300]
        }
    )
]

for _, _, params in models:
    params['scaler'] = [StandardScaler(), MinMaxScaler()]
