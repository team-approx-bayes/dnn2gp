import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class VectorizedNN(nn.Module):

    @property
    def gradient(self):
        grad = torch.cat([p.grad.data.flatten() for p in self.parameters()])
        return grad.numpy()

    @property
    def weights(self):
        wts = torch.cat([p.flatten() for p in self.parameters()])
        return wts

    @property
    def parameter_list(self):
        wts = [p.flatten() for p in self.parameters()]
        return wts

    def adjust_weights(self, weights):
        ix = 0
        for p in self.parameters():
            shape = p.data.shape
            if type(weights) is np.ndarray:
                p.data = torch.tensor(weights[ix:ix + np.prod(shape)]).view(*shape)
            else:
                p.data = weights[ix:ix + np.prod(shape)].view(*shape)
            ix += np.prod(shape)

    @property
    def transfer(self):
        if self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'relu':
            return F.relu
        elif self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'selu':
            return torch.nn.functional.selu


class SimpleMLP(VectorizedNN):
    def __init__(self, input_size, h_size=10, n_layers=3, activation='tanh', transfer_off=False):
        super().__init__()
        self.trans_off = transfer_off
        self.linear_model = (n_layers == 1)
        if n_layers == 1:  # linear model
            self.fc = nn.Linear(input_size, 1)
        else:
            self.fc_in = nn.Linear(input_size, h_size)  # fc_in, first layer
            self.hidden_layers = nn.ModuleList([nn.Linear(h_size, h_size) for _ in range(n_layers - 2)])
            self.fc_out = nn.Linear(h_size, 1)  # f_out, last layer
        self.activation = activation

    def forward(self, x):
        if self.linear_model:
            return self.fc(x)
        if self.trans_off:
            x = self.fc_in(x)
        else:
            x = self.transfer(self.fc_in(x))
        for layer in self.hidden_layers:
            x = self.transfer(layer(x))
        return self.fc_out(x)

    @property
    def transfer(self):
        if self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'relu':
            return F.relu
        elif self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'elu':
            return F.elu


class FMLP(VectorizedNN):
    # for now fixed to output size 1
    def __init__(self, input_size, hidden_sizes, activation='tanh', transfer_off=True):
        super().__init__()
        assert len(hidden_sizes) >= 1
        self.trans_off = transfer_off
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.fc_in = nn.Linear(input_size, hidden_sizes[0])
        if len(hidden_sizes) > 1:
            in_outs = zip(hidden_sizes[:-1], hidden_sizes[1:])
            self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size)
                                                for in_size, out_size in in_outs])
        else:
            self.hidden_layers = list()
        self.fc_out = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):
        if self.trans_off:
            x = self.fc_in(x)
        else:
            x = self.transfer(self.fc_in(x))
        for layer in self.hidden_layers:
            x = self.transfer(layer(x))
        return self.fc_out(x)


class SimpleConvNet(VectorizedNN):
    def __init__(self, input_size, h_size=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.n_after_conv = int((input_size / 4) * (input_size / 4) * 10)
        self.fc = nn.Linear(self.n_after_conv, 1)
        self.hw = input_size

    def forward(self, x):
        x = x.reshape(-1, 1, self.hw, self.hw)
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.n_after_conv)
        x = self.fc(x)
        return x


class WilliamsNN(VectorizedNN):
    """Neural Network with sigmoidal transfer according to williams 1997 Theorem,
    which considers below 1-hidden layer network with infinite width. Here, we have
    the finite couter-part."""

    def __init__(self, input_size, h_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, h_size)
        self.fc2 = nn.Linear(h_size, 1)

    def forward(self, x):
        x = torch.erf(self.fc1(x))
        x = self.fc2(x)
        return x

    @property
    def U(self):
        return torch.cat(self.parameter_list[:2])

    @property
    def V(self):
        return self.parameter_list[2]

    @property
    def b(self):
        return self.parameter_list[3]


class LeNet5(nn.Module):
    def __init__(self, input_channels=1, dims=28, num_classes=2):
        super(type(self), self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        dims = int((dims-4)/2)
        self.fc1 = nn.Linear(16*dims*dims, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc(out)
        return out


class LeNet5CIFAR(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
