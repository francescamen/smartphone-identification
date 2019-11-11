
"""
    encoder: defines the architecture of the GRU-based RNN encoder

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

import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.GRU(1, hidden_size, num_layers)
        self.drop = nn.Dropout(p=0.2)

    def encode(self, x):
        _, hidden = self.encoder(x)
        hidden = self.drop(hidden)
        return hidden

    def forward(self, x):
        hidden = self.encode(x)
        return hidden
