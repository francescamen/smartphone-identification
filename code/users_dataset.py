
"""
    users_dataset: defines the dataset structure for the autoencoder

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

from torch.utils.data import Dataset


class UsersDatasetAutoencoder(Dataset):

    def __init__(self, input_list):
        self.sentences = input_list

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = self.sentences[idx]
        return sample
