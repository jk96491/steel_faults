import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.model_infos = args.model_infos
        self.decoder_dims = self.model_infos["decoder_dims"]
        self.Decoder_Layers = nn.ModuleList()
        self.drop_out = self.model_infos["drop_out"]
        self.input_dim = args.input_dim

        cur_dim = self.model_infos["latent_dim"]
        for i in range(len(self.decoder_dims)):
            self.Decoder_Layers.append(nn.Sequential(nn.Linear(cur_dim, self.decoder_dims[i]),
                                                     nn.ReLU(),
                                                     nn.Dropout(self.drop_out)))
            cur_dim = self.decoder_dims[i]

        self.Decoder_Layers.append(nn.Linear(cur_dim, self.input_dim))

    def forward(self, latent):
        for layer in self.Decoder_Layers[:-1]:
            x = layer(latent)
        out = self.Decoder_Layers[-1](x)

        return out