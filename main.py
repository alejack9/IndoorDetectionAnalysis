from sklearn.model_selection import train_test_split

import data_layer
import model_runner
import preprocessing

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X, y, num_classes = data_layer.load_data()

    # preprocessing.priori_analysis(X, y)

    # create N different sub-datasets
    X_subsets = preprocessing.create_datasets(X)

    best_models = {}
    # for each sub-dataset
    for X_current in X_subsets:
        # 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(X_current, y, test_size=0.20, random_state=42, stratify=y)
        # replace missing values
        X_train, X_test = preprocessing.remove_nan(X_train, X_test)

        current_bests = model_runner.retrieve_best_models(X_train, y_train)
        print(current_bests)
