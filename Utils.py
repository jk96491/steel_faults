import numpy as np
import torch
import torch.nn.functional as F
import collections
from copy import deepcopy
import yaml


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


def normalize(data, exclude_cols_):
    for i in range(len(data[0])):
        if exclude_cols_.__contains__(i):
            continue

        cur_data = data[:, i]
        min, max = cur_data.min() , cur_data.max()
        normalize_data = (cur_data - min) / (max - min)
        data[:, i] = normalize_data


def standardization(data):
    for i in range(len(data[0])):
        cur_data = data[:, i]
        mean, std = np.mean(cur_data), np.std(cur_data)
        standardization_data = (cur_data - mean) / std
        data[:, i] = standardization_data

    return data


def test(test_len, model, x_test, y_test):
    correct_count = 0
    for i in range(test_len):
        result = torch.argmax(F.softmax(model(x_test[i])))
        answer = torch.argmax(y_test[i])

        if result.item() == answer.item():
            correct_count += 1

    print('accuracy : {0}'.format(correct_count / test_len))


def generate_data(x_data, x_train, y_train, auto_encoder):
    for i in range(20):
        generate_x_data = auto_encoder(x_data).detach()
        x_train = torch.cat([x_train, generate_x_data], dim=0)
        y_train = torch.cat([y_train, y_train], dim=0)

    return x_train, y_train


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_config(algorithm):
    config_dir = '{0}/{1}'
    config_dir2 = '{0}/{1}/{2}'

    with open(config_dir.format('config', "{}.yaml".format('default')), "r") as f:
        try:
            default_config = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    with open(config_dir2.format('config', 'args', "{}.yaml".format(algorithm)), "r") as f:
        try:
            config_dict2 = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format('sc2', exc)
        alg_config = config_dict2

    final_config_dict = recursive_dict_update(default_config, alg_config)

    return final_config_dict


