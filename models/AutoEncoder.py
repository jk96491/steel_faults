import torch.nn as nn
import torch.optim as optim
from models.Encoder import Encoder
from models.Decoder import Decoder


class AutoEncoder(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args

        self.model_infos = args.model_infos
        self.latent_dim = self.model_infos["encoder_dims"][-1]
        self.latent_sample_dim = self.model_infos["latent_dim"]

        self.encoder = Encoder(self.args)
        self.decoder = Decoder(self.args)

        self.optimizer = optim.Adam(self.parameters(), lr=self.model_infos["learning_rate"])
        self.device = device
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)

        return out

    def learn(self, input):
        predict = self.forward(input)
        loss = self.mse_loss(predict, input)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()