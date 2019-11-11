
"""
    plot_autoencoder: tests the autoencoder performance and plots the results

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
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
from encoder_rnn import RNNEncoder
from fully_connected import FullyConnected
from decoder_rnn import RNNDecoder
from users_dataset import UsersDatasetAutoencoder
import matplotlib.gridspec as gridspec
import pickle
matplotlib.use('TkAgg')


def tensor_from_sentence(sentence):
    sent_list = list(sentence)
    return torch.tensor(sent_list).view(-1, 1).float()


def collate_fn(_list):
    _list.sort(key=lambda x: x[0].shape[0], reverse=True)
    return _list


def input_packing(_list):
    tensor_list = [tensor_from_sentence(_list[i]) for i in range(len(_list))]
    packed_list = pack_sequence(tensor_list)
    return packed_list


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
    encoder_model = RNNEncoder(args.hidden_neurons, args.layers)
    fully_connect = FullyConnected(args.hidden_neurons)
    decoder_model = RNNDecoder(args.hidden_neurons, args.layers)

    encoder_model.load_state_dict(torch.load('../model_parameters/encoder_model_' + str(num_traces) + '.pt'))
    fully_connect.load_state_dict(torch.load('../model_parameters/fully_connected_' + str(num_traces) + '.pt'))
    decoder_model.load_state_dict(torch.load('../model_parameters/decoder_model_' + str(num_traces) + '.pt'))

    tensor_sentences_train = [[tensor_from_sentence(sentences_train[i]), tensor_from_sentence(mInfo_train[i][7:8] - 1),
                               tensor_from_sentence(mTime_train[i])] for i in range(len(sentences_train))]
    users_dataset_train = UsersDatasetAutoencoder(tensor_sentences_train)

    tensor_sentences_test = [[tensor_from_sentence(sentences_test[i]),
                              tensor_from_sentence(mInfo_test[i][7:8] - 1),
                              tensor_from_sentence(mTime_test[i])] for i in range(len(sentences_test))]
    users_dataset_test = UsersDatasetAutoencoder(tensor_sentences_test)

    batch_size = 1
    train_loader = DataLoader(users_dataset_train, batch_size=batch_size, shuffle=False, num_workers=0,
                              collate_fn=collate_fn)

    test_loader = DataLoader(users_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
                             collate_fn=collate_fn)

    fig = plt.figure(1)
    gridspec.GridSpec(2, 1)
    plt.subplot2grid((2, 1), (0, 0))

    index_last = 0
    for i in range(50, 60):
        sample_batched = [users_dataset_test[i]]
        sample_batched_sent = [sample_batched[i][0] for i in range(len(sample_batched))]
        if sample_batched_sent[0].shape[0] > 0:
            sample_batched_labels = [sample_batched[i][1] for i in range(len(sample_batched))]
            sample_batched_time = [sample_batched[i][2][1] for i in range(len(sample_batched))]
            sample_batched_duration = [sample_batched[i][2][0]*10 for i in range(len(sample_batched))]

            sample_batched_sent = torch.stack(sample_batched_sent)
            sample_batched_sent = torch.transpose(sample_batched_sent, 0, 1)
            x_input = sample_batched_sent
            x_input = torch.reshape(x_input, (x_input.shape[0], 1, 1))
            hidden_layer = fully_connect(encoder_model(x_input))
            representation = torch.transpose(hidden_layer, 0, 1)
            representation = representation.view(1, -1)

            dec_out = [torch.zeros(1, 1, 1)]
            out_vector = torch.zeros(sample_batched_sent.shape[0], 1)
            dec_hidden = hidden_layer[:, :, :]
            for word in range(out_vector.shape[0]):
                dec_out = input_packing(dec_out)
                out, dec_hidden = decoder_model(dec_out, dec_hidden)
                dec_out = out
                out_vector[word] = dec_out

            out_vector_array = pad_sequence(out_vector).data[0].numpy()
            x_input_array = x_input.data.numpy()[:, 0, 0]

            x_ax = np.linspace(index_last,
                               out_vector_array.shape[0] + index_last,
                               out_vector_array.shape[0])
            index_last = index_last + out_vector_array.shape[0]

            plt.plot(x_ax, out_vector_array*20000, c='firebrick', marker='x',  linestyle='--', markersize=3,
                     linewidth=0.8)
            plt.plot(x_ax, x_input_array*20000, c='midnightblue', marker='o',  markersize=3, linewidth=0.8)
            plt.axvline(x=index_last + 1, color='k', linewidth=2, linestyle='--')
            index_last = index_last + 1 + 1
    plt.grid()
    plt.legend(('predicted', 'actual'), loc='upper right')
    plt.ylabel('Character value', FontSize=12)
    plt.xlabel('Word time slot', FontSize=12)
    plt.xticks(FontSize=12)
    plt.yticks(FontSize=12)

    plt.subplot2grid((2, 1), (1, 0))
    index_last = 0
    for i in range(900, 910):
        sample_batched = [users_dataset_test[i]]
        sample_batched_sent = [sample_batched[i][0] for i in range(len(sample_batched))]
        if sample_batched_sent[0].shape[0] > 0:
            sample_batched_labels = [sample_batched[i][1] for i in range(len(sample_batched))]
            sample_batched_time = [sample_batched[i][2][1] for i in range(len(sample_batched))]
            sample_batched_duration = [sample_batched[i][2][0]*10 for i in range(len(sample_batched))]

            sample_batched_sent = torch.stack(sample_batched_sent)
            sample_batched_sent = torch.transpose(sample_batched_sent, 0, 1)
            x_input = sample_batched_sent
            x_input_array = x_input.data.numpy()
            x_input = torch.reshape(x_input, (x_input.shape[0], 1, 1))
            hidden_layer = fully_connect(encoder_model(x_input))
            representation = torch.transpose(hidden_layer, 0, 1)
            representation = representation.view(1, -1)
            representation_array = representation.data.numpy()

            dec_out = [torch.zeros(1, 1, 1)]
            out_vector = torch.zeros(sample_batched_sent.shape[0], 1)
            dec_hidden = hidden_layer[:, :, :]
            for word in range(out_vector.shape[0]):
                dec_out = input_packing(dec_out)
                out, dec_hidden = decoder_model(dec_out, dec_hidden)
                dec_out = out
                out_vector[word] = dec_out

            out_vector_array = pad_sequence(out_vector).data[0].numpy()
            x_input_array = x_input.data.numpy()[:, 0, 0]

            x_ax = np.linspace(index_last,
                               out_vector_array.shape[0] + index_last,
                               out_vector_array.shape[0])
            index_last = index_last + out_vector_array.shape[0]

            mk = matplotlib.markers.MarkerStyle(marker='.', fillstyle='none')
            plt.plot(x_ax, out_vector_array*20000, c='firebrick', marker='x', linestyle='--',  markersize=3,
                     linewidth=0.8)
            plt.plot(x_ax, x_input_array*20000, c='midnightblue', marker='o',  markersize=3, linewidth=0.8)
            plt.axvline(x=index_last + 1, color='k', linewidth=2, linestyle='--')
            index_last = index_last + 1 + 1
    plt.grid()
    plt.legend(('predicted', 'actual'), loc='upper right')
    plt.ylabel('Character value', FontSize=12)
    plt.xlabel('Word time slot', FontSize=12)
    plt.xticks(FontSize=12)
    plt.yticks(FontSize=12)

    fig.tight_layout()
    fig.set_size_inches(w=11, h=4)
    fig.savefig('../plots/autoencoder_multiple.eps')
