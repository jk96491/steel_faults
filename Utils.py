import numpy as np
import torch

def SuffleData(x_train, y_train, count):
    s = np.arange(x_train.shape[0])
    np.random.shuffle(s)

    x_train = x_train[s]
    y_train = y_train[s]

    x_train = x_train[:count]
    y_train = y_train[:count]

    return x_train, y_train


def convertToTensorInput(input, input_size):
    input = np.reshape(input, [1, input_size])
    return torch.FloatTensor(input)


def normalize(data):
    for i in range(len(data[0])):

        if i == 11 or i ==12 or i == 19 or i == 20 or i ==26:
            continue

        cur_data = data[:, i]
        min, max = cur_data.min() , cur_data.max()
        normalize_data = (cur_data - min) / (max - min)
        data[:, i] = normalize_data

    return data


def standardization(data):
    for i in range(len(data[0])):
        cur_data = data[:, i]
        mean, std = np.mean(cur_data), np.std(cur_data)
        standardization_data = (cur_data - mean) / std
        data[:, i] = standardization_data

    return data



