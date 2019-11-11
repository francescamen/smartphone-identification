
"""
    fully_connected: defines a fully connected layer with tanh activation function

    Copyright (C) 2019 Francesca Meneghello, Michele Rossi, Nicola Bui
    contact: meneghello@dei.unipd.it

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, hidden_size):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.tan = nn.Tanh()

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        connect = self.fc(x)
        connect = self.tan(connect)
        connect = torch.transpose(connect, 0, 1)
        return connect
