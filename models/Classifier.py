import torch
import torch.nn as nn
from Utils import SuffleData
from Utils import test
import torch.nn.functional as F
import torch.optim as optim


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.Layers = nn.ModuleList()
        self.hidden_dims = self.args.hidden_dims
        self.drop_out = self.args.drop_out

        cur_dim = self.input_dim
        for i in range(len(self.hidden_dims)):
            self.Layers.append(nn.Sequential(nn.Linear(cur_dim, self.hidden_dims[i]),
                                                     nn.ReLU(),
                                                     nn.Dropout(self.drop_out)))
            cur_dim = self.hidden_dims[i]

        self.Layers.append(nn.Linear(cur_dim, self.args.n_class))
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)

    def forward(self, x):
        for layer in self.Layers[:-1]:
            x = layer(x)
        out = self.Layers[-1](x)

        return out

    def learn(self, x_train, y_train, test_len, model, x_test, y_test, batch_size):
        nb_epochs = 10000

        for epoch in range(nb_epochs + 1):
            x_train, y_train = SuffleData(x_train, y_train, batch_size)

            hypothesis = self.forward(x_train)
            output = torch.max(y_train, 1)[1]
            cost = F.cross_entropy(hypothesis, output)

            self.optimizer.zero_grad()
            cost.backward()
            self.optimizer.step()

            # 20번마다 로그 출력
            if epoch % self.args.log_interval == 0:
                print('Epoch {:4d}/{} Cost: {:.6f} '.format(epoch, nb_epochs, cost.item()))

            if epoch % self.args.test_interval == 0 and epoch is not 0:
                test(test_len, model, x_test, y_test)