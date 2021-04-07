import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Utils import SuffleData
from Utils import normalize

batch_size = 1024
training_len = 1800
use_normalize = True
drop_out = 0.1
drop_out_AE = 0.1
exclude_cols = [11, 12, 19, 20, 26]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_layer_1 = nn.Sequential(nn.Linear(27, 16),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out_AE))
        self.encoder_layer_2 = nn.Sequential(nn.Linear(16, 8),
                                             nn.ReLU(),
                                             nn.Dropout(drop_out_AE))
        self.decoder_layer_1 = nn.Sequential(nn.Linear(8, 16),
                                      nn.ReLU(),
                                      nn.Dropout(drop_out_AE))
        self.decoder_layer_2 = nn.Sequential(nn.Linear(16, 27),
                                             nn.ReLU(),
                                             nn.Dropout(drop_out_AE))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def encoder(self, x):
        x = self.encoder_layer_1(x)
        x = self.encoder_layer_2(x)

        return x

    def decoder(self, x):
        x = self.decoder_layer_1(x)
        x = self.decoder_layer_2(x)

        return x


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(8, 64),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out))
        self.layer2 = nn.Sequential(nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out))
        self.layer3 = nn.Sequential(nn.Linear(32, 16),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out))
        self.layer4 = nn.Sequential(nn.Linear(16, 7))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

xy = np.loadtxt('faults.csv', delimiter=',') #1941
x_data = xy[:, :-7]
y_data = xy[:, -7:]

x_data = torch.FloatTensor(x_data).to(device)
y_data = torch.FloatTensor(y_data).to(device)

if use_normalize:
    normalize(x_data, exclude_cols)

def train_autoEncoder(x_data, y_data):
    auto_encoder = AutoEncoder().to(device)

    nb_epochs = 10000

    optimizer = optim.Adam(auto_encoder.parameters(), lr=0.001)

    for epoch in range(nb_epochs):
        x_train, y_train = SuffleData(x_data, y_data, batch_size)

        predict = auto_encoder(x_train)
        cost = F.mse_loss(predict, x_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f} '.format(
                epoch, nb_epochs, cost.item(),
            ))

    return auto_encoder

def train_NN(x_data, y_data, auto_encoder):
    model = Classifier().to(device)

    x_data, y_data = SuffleData(x_data, y_data, len(xy))

    x_train = x_data[:training_len]
    y_train = y_data[:training_len]

    x_test = x_data[training_len:]
    y_test = y_data[training_len:]
    test_len = len(x_test)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    nb_epochs = 10000

    for epoch in range(nb_epochs + 1):
        x_train, y_train = SuffleData(x_train, y_train, batch_size)

        hypothesis = model(auto_encoder.encoder(x_train))
        output = torch.max(y_train, 1)[1]
        cost = F.cross_entropy(hypothesis, output)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 20번마다 로그 출력
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f} '.format(
                epoch, nb_epochs, cost.item(),
            ))

        # 100번 마다 Test
        if epoch % 1000 == 0 and epoch is not 0:
            correct_count = 0
            for i in range(test_len):
                result = torch.argmax(F.softmax(model(auto_encoder.encoder(x_test[i]))))
                answer = torch.argmax(y_test[i])

                if result.item() == answer.item():
                    correct_count += 1

            print('accuracy : {0}'.format(correct_count / test_len))


auto_encoder = train_autoEncoder(x_data, y_data)
train_NN(x_data, y_data, auto_encoder)