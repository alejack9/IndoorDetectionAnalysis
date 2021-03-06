import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from os import path
import preprocessing
from pytorch.model_runner import train_loop, test_loop
import torch
from pytorch.model import Feedforward
from pytorch.dataset import IndoorDataset
from torch.utils.data import DataLoader, Subset
import itertools
import time


# starting point for neural network training and testing
def run(X, y, nn_models_dir, use_saved_if_available, save_models):
    # hyperparameters
    hidden_sizes = np.concatenate([
        [128, 64, 32],  # , 8],
        [256, 100, 50, 16]])
    nums_epochs = [1000, 500]
    batch_sizes = [16, 32, 64, 128, 256]
    gamma = [0.02, 0.05, 0.08, 0.1]
    optimizerBuilders = [torch.optim.SGD]
    learning_rate = 0.1

    # exploit gpu if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Device: {device}')
    fs = X.shape[1]

    model_file = path.join(nn_models_dir, f'NN_{fs}.torch')
    result_file = path.join(nn_models_dir, 'csvs', f'NN_{fs}.csv')

    # cartesian product of the hyperparameters' sets
    hyperparams = itertools.product(
        hidden_sizes, nums_epochs, batch_sizes, gamma, optimizerBuilders)

    # split the indices (72% train, 8% val, 20% test)
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    # remove missing values
    X[train_idx], X[test_idx] = preprocessing.remove_nan(
        X[train_idx], X[test_idx])
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.1, stratify=y[train_idx], random_state=42)

    # Min-Max Scaling
    scaler = MinMaxScaler()
    scaler.fit(X[train_idx])
    X[train_idx] = scaler.transform(X[train_idx])
    X[val_idx] = scaler.transform(X[val_idx])
    X[test_idx] = scaler.transform(X[test_idx])

    # save the dataset as a set of tensors
    dataset = IndoorDataset(X, y)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    best_val_score = 0
    losses = None
    if not use_saved_if_available or not path.exists(model_file):
        # validation of hyperparameters
        for hidden_size, num_epochs, batch_size, gamma, optimizerBuilder in hyperparams:
            print('---------------------------------------------------------------')
            print(
                f'Dataset size: {fs}, hidden_size: {hidden_size}, num_epochs: {num_epochs}, batch_size: {batch_size}, gamma: {gamma}, optimizer: {optimizerBuilder}')

            # create dataloaders to be passed to train_loop
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
            test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

            # create the neural network
            model = Feedforward(
                dataset.X.shape[1], hidden_size, dataset.num_classes)
            model.to(device)
            criterion = torch.nn.CrossEntropyLoss()

            # stochastic gradient descent
            optimizer = optimizerBuilder(model.parameters(), lr=learning_rate)

            # scheduler fo learning rate decay
            def lambda1(epoch): return 1 / (1 + gamma * epoch)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda1)

            # train the neural network
            time_before = time.time()
            losses = train_loop(train_loader, model, criterion,
                                optimizer, scheduler, num_epochs, device)
            time_after_train = time.time() - time_before

            # evaluate the neural network on validation and test set
            train_score = test_loop(DataLoader(
                train_subset, batch_size=1, shuffle=False), model, device)
            time_before = time.time()
            val_score = test_loop(val_loader, model, device)
            time_after_val = time.time() - time_before
            test_score = test_loop(test_loader, model, device)

            print(
                f'Dataset size: {fs}, hidden_size: {hidden_size}, num_epochs: {num_epochs}, batch_size: {batch_size}, gamma: {gamma}')
            print(f"Train accuracy: {train_score}")
            print(f"Validation accuracy: {val_score}")

            # save the neural network if it has the best validation score until now
            if val_score > best_val_score:
                best_model = {"NN_" + str(fs): {"pipeline": "mlp",
                                                "hidden_size": hidden_size,
                                                "epochs": num_epochs,
                                                "batch_size": batch_size,
                                                "decay": gamma,
                                                "mean_train_score": train_score,
                                                "mean_test_score": val_score,
                                                "mean_fit_time": time_after_train,
                                                "mean_score_time": time_after_val,
                                                "final_test_score": test_score,
                                                "losses": losses}}
                best_val_score = val_score
                best_nn = model

        # save on disk the best neural network
        if save_models:
            pd.DataFrame(best_model).transpose().to_csv(result_file)
            torch.save(best_nn.state_dict(), model_file)
    else:
        # load the best neural network
        print(f"Saved model found: NN_{fs}")
        result = pd.read_csv(result_file, index_col=0)
        model = Feedforward(
            dataset.X.shape[1], result['hidden_size'][0], dataset.num_classes)
        model.load_state_dict(torch.load(model_file), strict=False)
        model.to(device)
        best_model = result.transpose().to_dict()
        losses = np.array(best_model["NN_" + str(fs)]
                          ["losses"][1: -1].split(",")).astype(np.float)

    return best_model, losses
