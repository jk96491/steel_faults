import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Utils import SuffleData
from Utils import normalize

batch_size = 2048
training_len = 1500
use_normalize = True
drop_out = 0.25
drop_out_AE = 0.25
exclude_cols = [11, 12, 19, 20, 26]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = nn.Sequential(nn.Linear(27, 16),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out_AE))
        self.encoder2 = nn.Sequential(nn.Linear(16, 8),
                                      nn.ReLU(),
                                      nn.Dropout(drop_out_AE))
        self.encoder3 = nn.Sequential(nn.Linear(8, 4),
                                      nn.ReLU(),
                                      nn.Dropout(drop_out_AE))

        self.decoder1 = nn.Sequential(nn.Linear(4, 8),
                                      nn.ReLU(),
                                      nn.Dropout(drop_out_AE))
        self.decoder2 = nn.Sequential(nn.Linear(8, 16),
                                      nn.ReLU(),
                                      nn.Dropout(drop_out_AE))
        self.decoder3 = nn.Sequential(nn.Linear(16, 27))

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)

        return x

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(27, 512),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out))
        self.layer2 = nn.Sequential(nn.Linear(512, 128),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out))
        self.layer3 = nn.Sequential(nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out))
        self.layer4 = nn.Sequential(nn.Linear(64, 16),
                                    nn.ReLU(),
                                    nn.Dropout(drop_out))
        self.layer5 = nn.Sequential(nn.Linear(16, 7))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
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

    optimizer = optim.Adam(auto_encoder.parameters(), lr=0.0005)

    for epoch in range(nb_epochs):
        cur_x_train, cur_y_train = SuffleData(x_data, y_data, 512)

        predict = auto_encoder(cur_x_train)
        cost = F.mse_loss(predict, cur_x_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f} '.format(
                epoch, nb_epochs, cost.item(),
            ))

    return auto_encoder


def generate_data(x_train, y_train, auto_encoder):
    for i in range(10):
        generate_x_data = auto_encoder(x_data).detach()
        x_train = torch.cat([x_train, generate_x_data], dim=0)
        y_train = torch.cat([y_train, y_train], dim=0)

    return x_train, y_train


def train_NN(x_data, y_data, auto_encoder):
    model = Classifier().to(device)

    x_data, y_data = SuffleData(x_data, y_data, len(xy))

    x_train = x_data[:training_len]
    y_train = y_data[:training_len]

    x_train, y_train = generate_data(x_train, y_train, auto_encoder)

    x_test = x_data[training_len:]
    y_test = y_data[training_len:]
    test_len = len(x_test)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    nb_epochs = 10000

    for epoch in range(nb_epochs + 1):
        cur_x_train, cur_y_train = SuffleData(x_train, y_train, batch_size)

        hypothesis = model(cur_x_train)
        output = torch.max(cur_y_train, 1)[1]
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
                result = torch.argmax(F.softmax(model(x_test[i])))
                answer = torch.argmax(y_test[i])

                if result.item() == answer.item():
                    correct_count += 1

            print('accuracy : {0}'.format(correct_count / test_len))


auto_encoder = train_autoEncoder(x_data, y_data)
train_NN(x_data, y_data, auto_encoder)