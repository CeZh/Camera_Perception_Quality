import torch.nn as nn
import torch

class MLP_Regressor(nn.Module):
    def __init__(self, configs, output_dim):
        super(MLP_Regressor, self).__init__()
        self.superpixel = configs['model_parameters']['use_superpixel']
        self.linear1 = nn.Linear(output_dim, 512)
        if self.superpixel:
            self.linear2 = nn.Linear(512, int(configs['superpixel_parameters']['att_dim']))
            self.linear3 = nn.Linear(int(configs['superpixel_parameters']['att_dim']*2),
                                     int(configs['superpixel_parameters']['att_dim']*2))
            self.linear4 = nn.Linear(int(configs['superpixel_parameters']['att_dim'] * 2), 1)
        else:
            self.linear2 = nn.Linear(512, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        if self.superpixel:
            super_x = x['super_img']
            x = x['img']
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        out = self.linear2(x)
        if self.superpixel:
            out = torch.cat((out, super_x), dim=1)
            out = self.linear3(out)
            out = self.dropout(out)
            out = self.gelu(out)
            out = self.linear4(out)
        return out
