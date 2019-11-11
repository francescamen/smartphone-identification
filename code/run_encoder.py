
"""
    run_encoder: outputs the code for each input word using the GRU-based RNN encoder

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

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from encoder_rnn import RNNEncoder
from fully_connected import FullyConnected
from decoder_rnn import RNNDecoder
import pickle
plt.switch_backend('agg')
device = torch.device("cuda")


def tensor_from_sentence(sentence):
    sent_list = list(sentence)
    return torch.tensor(sent_list).view(-1, 1).float()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hidden_neurons', help='Number of hidden neurons', type=int)
    parser.add_argument('layers', help='Number of layers', type=int)
    args = parser.parse_args()

    num_traces = 40

    with open('../processed_files/mInfo_train.txt', "rb") as fp:  # Unpickling
        mInfo_train = pickle.load(fp)
    with open('../processed_files/mInfo_test.txt', "rb") as fp:  # Unpickling
        mInfo_test = pickle.load(fp)

    with open('../processed_files/mTime_train.txt', "rb") as fp:  # Unpickling
        mTime_train = pickle.load(fp)
    with open('../processed_files/mTime_test.txt', "rb") as fp:  # Unpickling
        mTime_test = pickle.load(fp)

    with open('../processed_files/sentences_train.txt', "rb") as fp:  # Unpickling
        sentences_train = pickle.load(fp)
    with open('../processed_files/sentences_test.txt', "rb") as fp:  # Unpickling
        sentences_test = pickle.load(fp)

    # ---------------------------------------------------------------------------------------------------------------- #
    # CODE THE SENTENCES #

    encoder_model = RNNEncoder(args.hidden_neurons, args.layers).to(device)
    fully_connect = FullyConnected(args.hidden_neurons).to(device)
    decoder_model = RNNDecoder(args.hidden_neurons, args.layers).to(device)

    encoder_model.load_state_dict(torch.load('../model_parameters/encoder_model_' + str(num_traces) + '.pt'))
    fully_connect.load_state_dict(torch.load('../model_parameters/fully_connected_' + str(num_traces) +
                                             '.pt'))
    decoder_model.load_state_dict(torch.load('../model_parameters/decoder_model_' + str(num_traces) + '.pt'))

    mTime_train = np.asarray(mTime_train)
    mTime_test = np.asarray(mTime_test)
    mInfo_train = np.asarray(mInfo_train)
    mInfo_test = np.asarray(mInfo_test)

    for trace in range(num_traces):
        indices = np.argwhere(mInfo_train[:, 7] == trace + 1)[:, 0]
        sentences_single_user_train = sentences_train[indices[0]:indices[-1]]

        hiddens = torch.zeros((len(sentences_single_user_train), args.hidden_neurons*args.layers), device=device)
        for i_w in range(len(sentences_single_user_train)):
            x_input = tensor_from_sentence(sentences_single_user_train[i_w])
            x_input = torch.reshape(x_input, (x_input.shape[0], 1, 1)).to(device)
            representation = fully_connect(encoder_model(x_input))
            representation = torch.transpose(representation, 0, 1).contiguous().to(device)
            representation = representation.view(1, -1)
            hiddens[i_w, :] = representation

        with open('../processed_files/hiddens_train_user' + str(trace) + '.txt', "wb") as fp:  # Pickling
            pickle.dump(hiddens, fp)
        with open('../processed_files/times_train_user' + str(trace) + '.txt', "wb") as fp:  # Pickling
            pickle.dump(mTime_train[indices[0]:indices[-1], :], fp)

        indices = np.argwhere(mInfo_test[:, 7] == trace + 1)[:, 0]
        sentences_single_user_test = sentences_test[indices[0]:indices[-1]]

        hiddens = torch.zeros((len(sentences_single_user_test), args.hidden_neurons*args.layers))
        for i_w in range(len(sentences_single_user_test)):
            x_input = tensor_from_sentence(sentences_single_user_test[i_w])
            x_input = torch.reshape(x_input, (x_input.shape[0], 1, 1)).to(device)
            representation = fully_connect(encoder_model(x_input))
            representation = torch.transpose(representation, 0, 1).contiguous().to(device)
            representation = representation.view(1, -1)
            hiddens[i_w, :] = representation

        with open('../processed_files/hiddens_test_user' + str(trace) + '.txt', "wb") as fp:  # Pickling
            pickle.dump(hiddens, fp)
        with open('../processed_files/times_test_user' + str(trace) + '.txt', "wb") as fp:  # Pickling
            pickle.dump(mTime_test[indices[0]:indices[-1], :], fp)
