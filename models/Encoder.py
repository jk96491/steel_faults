import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.model_infos = args.model_infos
        self.encoder_dims = self.model_infos["encoder_dims"]
        self.Encoder_Layers = nn.ModuleList()
        self.drop_out = self.model_infos["drop_out"]
        self.input_dim = args.input_dim

        cur_dim = self.input_dim
        for i in range(len(self.encoder_dims)):
            self.Encoder_Layers.append(nn.Sequential(nn.Linear(cur_dim, self.encoder_dims[i]),
                                                     nn.ReLU(),
                                                     nn.Dropout(self.drop_out)))
            cur_dim = self.encoder_dims[i]

    def forward(self, x):
        for layer in self.Encoder_Layers[:-1]:
            x = layer(x)
        latent = self.Encoder_Layers[-1](x)

        return latent