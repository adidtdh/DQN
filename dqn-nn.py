import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

#I am one of those fools that bought an AMD Gpu and it doesnt work on Linux so I am using windows :(
import torch_directml
dml = torch_directml.device()

class DeepQNetwork(nn.Module):
    # fc_dims is list of the nodes in a layer
    def __init__(self, lr, in_dims, out_dims, fc_dims) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.fc_dims = fc_dims

        self.fc_dims.insert(0, self.in_dims)
        self.fc_dims.append(self.out_dims)

        self.layers = []

        # trying this stupid thing out
        for i in range(len(self.fc_dims)-1):
            self.layers.append(nn.Linear(self.fc_dims[i], self.fc_dims[i+1]))

            if i != len(self.fc_dims)-1:
                self.layers.append(F.relu())
        
        self.layers.append(F.sigmoid()) # get values between 0 and 1 cuz 0 means sell and 1 means buy :)

        # stupid way to make a modules list and then unpack into sequential
        self.layers = nn.Sequential(*nn.ModuleList(self.layers))
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss() # this is not going to work and will need to be changed
        self.device = dml
        self.to(self.device) # benifit of putting everyithing in this model is that it is easier to send the entire thing to the gpu
    
    def forward(self, input):
        self.layers(input)




