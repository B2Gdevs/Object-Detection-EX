"""
Author: Benjamin Garrard

Conform every model that should be made to accept input_neurons, hidden_neurons,
and output_neurons.  That way the pytorch pipeline won't break and whomever
is running the experiments will have an easier time.
"""

import torch
import torch.nn as nn


class hdf_Dataset(torch.utils.data.Dataset):
    """
    Just the dataset class to batch more easily for pytorch.

    Honestly, this coud be given any two numpy arrays corresponding to
    X and Y data and this dataset would work for them.
    """

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return (self.x_data[index], self.y_data[index])

    def __len__(self):
        return self.y_data.shape[0]

class logic_net(nn.Module):
    def __init__(self, input_neurons, hidden, output_neurons): # all models must conform to input, hidden, and output to be used.
                                                               # interestingly as soon as ReLU is used the network immediately finds 
                                                               # a local minimum
        super(logic_net, self).__init__()
        self.fc1 = nn.Linear(input_neurons, output_neurons)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        torch.manual_seed(1)
        nn.init.uniform_(self.fc1.weight, -1, 1)
        nn.init.uniform_(self.fc1.bias, -1, 1)


    def forward(self, x):
        out = self.fc1(x)
        out = self.sig(out)
        return out

class deep_logic_net(nn.Module): # Relus will cause the network to find a local minimum
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super(deep_logic_net, self).__init__()
        self.fc1 = nn.Linear(input_neurons, hidden_neurons)
        self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, output_neurons)
        torch.manual_seed(1)
        nn.init.uniform_(self.fc1.weight, -1, 1)
        nn.init.uniform_(self.fc1.bias, -1, 1)
        nn.init.uniform_(self.fc2.weight, -1, 1)
        nn.init.uniform_(self.fc2.bias, -1, 1)
        nn.init.uniform_(self.fc3.weight, -1, 1)
        nn.init.uniform_(self.fc3.bias, -1, 1)



    def forward(self, x):
        out = self.fc1(x)
        out = self.sig(out)
        # out += x
        out = self.fc2(out)
        out = self.sig(out)
        out = self.fc3(out)
        out = self.sig(out)

        return out
    
