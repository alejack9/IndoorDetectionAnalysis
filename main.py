import data_layer
import preprocessing

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X, y, num_classes = data_layer.load_data()
    preprocessing.priori_analysis(X, y)
    # print(X)
    # print('-------------')
    # print(y)
    # print('-------------')
    # print(num_classes)
    # print(np.unique(y))