import torch
import numpy as np
from models.AutoEncoder import AutoEncoder
from types import SimpleNamespace as SN
from models.VAE import vae
from models.Classifier import Classifier
import Utils

algorithm = 'VAE'
config = Utils.config_copy(Utils.get_config(algorithm))
args = SN(**config)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

xy = np.loadtxt('Testing.csv', delimiter=',') #1941
x_data = xy[:, :-args.n_class]
y_data = xy[:, -args.n_class:]

x_data = torch.FloatTensor(x_data).to(device)
y_data = torch.FloatTensor(y_data).to(device)

if args.use_normalize:
    Utils.normalize(x_data, args.exclude_cols)


def train_model(x_data, y_data):
    model = None

    if args.name == "VAE":
        model = vae(args, device).to(device)
    elif args.name == "AutoEncoder":
        model = AutoEncoder(args, device).to(device)

    for epoch in range(args.nb_epochs):
        x_train, y_train = Utils.SuffleData(x_data, y_data, args.batch_size)

        loss = model.learn(x_train)

        if epoch % args.log_interval == 0:
            print('Epoch {:4d}/{} loss: {:.6f} '.format(epoch, args.nb_epochs, loss))

    return model


def train_classifier(x_data, y_data, auto_encoder):
    model = Classifier(args).to(device)

    x_data, y_data = Utils.SuffleData(x_data, y_data, len(xy))

    x_train = x_data[:args.training_len]
    y_train = y_data[:args.training_len]

    x_train, y_train = Utils.generate_data(x_data, x_train, y_train, auto_encoder)

    x_test = x_data[args.training_len:]
    y_test = y_data[args.training_len:]
    test_len = len(x_test)

    model.learn(x_train, y_train, test_len, model, x_test, y_test, args.batch_size)


train_classifier(x_data, y_data, train_model(x_data, y_data))