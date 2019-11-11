
"""
    decoder_rnn: defines the architecture of the GRU-based RNN decoder

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
from torch.nn.utils.rnn import pad_packed_sequence


class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.decoder = nn.GRU(1, hidden_size, num_layers)
        self.lin = nn.Linear(hidden_size, 1)

    def decode(self, x, hidden):
        decoder_output, hidden = self.decoder(x, hidden)
        decoder_output = pad_packed_sequence(decoder_output, batch_first=True)[0]
        decoder_output = torch.transpose(decoder_output, 0, 1)
        decoder_output = self.lin(decoder_output)
        decoder_output = torch.transpose(decoder_output, 0, 1)
        return decoder_output, hidden

    def forward(self, x, hidden):
        output, hidden = self.decode(x, hidden)
        return output, hidden
