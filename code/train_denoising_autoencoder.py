
"""
    train_denoising_autoencoder: trains the autoencoder and save the coding network parameters

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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from encoder_rnn import RNNEncoder
from fully_connected import FullyConnected
from decoder_rnn import RNNDecoder
from users_dataset import UsersDatasetAutoencoder
import pickle
import gc
plt.switch_backend('agg')
device = torch.device("cuda")


def tensor_from_sentence(sentence):
    return torch.tensor(list(sentence), device=device).view(-1, 1).float()


def collate_fn_autoencoder(_list):
    _list.sort(key=len, reverse=True)
    return _list


def collate_fn(_list):
    return _list


def input_packing(_list):
    tensor_list = [tensor_from_sentence(_list[i]) for i in range(len(_list))]
    return pack_sequence(tensor_list)


def target_packing(_list):
    tensor_list = [tensor_from_sentence(torch.cat((torch.zeros((1,), device=device), _list[i][:-1, 0]), 0)) for i in
                   range(len(_list))]
    packed_list = pack_sequence(tensor_list)
    return packed_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hidden_neurons', help='Number of hidden neurons', type=int)
    parser.add_argument('layers', help='Number of layers', type=int)
    parser.add_argument('epochs', help='Number of epochs', type=int)
    args = parser.parse_args()

    num_traces = 40

    with open('../processed_files/sentences_train.txt', "rb") as fp:  # Unpickling
        sentences_train = pickle.load(fp)
    with open('../processed_files/sentences_test.txt', "rb") as fp:  # Unpickling
        sentences_test = pickle.load(fp)

    # ---------------------------------------------------------------------------------------------------------------- #
    # AUTOENCODER #

    encoder_model = RNNEncoder(args.hidden_neurons, args.layers).to(device)
    fully_connect = FullyConnected(args.hidden_neurons).to(device)
    decoder_model = RNNDecoder(args.hidden_neurons, args.layers).to(device)

    print(encoder_model)
    print(fully_connect)
    print(decoder_model)

    criterion = nn.L1Loss(reduction='none')
    criterion_val = nn.L1Loss()

    parameters_dictionary = list(encoder_model.parameters()) + list(fully_connect.parameters()) + \
        list(decoder_model.parameters())

    optimizer = optim.Adam(parameters_dictionary, lr=0.001)

    tensor_sentences_train = [tensor_from_sentence(sentences_train[i]) for i in range(len(sentences_train))]
    users_dataset_train = UsersDatasetAutoencoder(tensor_sentences_train)

    tensor_sentences_test = [tensor_from_sentence(sentences_test[i]) for i in range(len(sentences_test))]
    users_dataset_test = UsersDatasetAutoencoder(tensor_sentences_test)

    batch_size = 64
    train_loader = DataLoader(users_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=collate_fn_autoencoder)

    test_loader = DataLoader(users_dataset_test, batch_size=batch_size, shuffle=True, num_workers=0,
                             collate_fn=collate_fn_autoencoder)

    max_epochs = args.epochs
    print_loss_total = 0
    print_every = 500
    print_every_val = 50
    iteration = 0
    for epoch in range(max_epochs):
        print('----epoch ' + str(epoch) + ' training----')
        for i_batch, sample_batched in enumerate(train_loader):
            iteration += 1
            x_input = input_packing(sample_batched)
            x_target = target_packing(sample_batched).to(device)

            erasure_prob = 0.1
            sample_batched_noise = []
            for ii in range(len(sample_batched)):
                random_mask_array = np.random.random(sample_batched[ii].shape[0])
                random_mask_array = random_mask_array > erasure_prob
                random_mask_array = random_mask_array.astype(int)
                random_mask = torch.tensor(random_mask_array, device=device).float().view(-1, 1)
                sample_batched_noise.append(sample_batched[ii] * random_mask)
            x_input_noise = input_packing(sample_batched_noise).to(device)

            encoder_model.zero_grad()
            fully_connect.zero_grad()
            decoder_model.zero_grad()

            hidden = encoder_model(x_input_noise).to(device)

            conn = fully_connect(hidden).contiguous().to(device)

            out, dec_hidden = decoder_model(x_target, conn)

            output_unpacked = out

            input_unpacked = pad_packed_sequence(x_input, batch_first=True)
            mask = torch.clone(input_unpacked[0])
            mask[mask > 0] = 1
            output_unpacked = output_unpacked * mask

            loss = criterion(output_unpacked, input_unpacked[0])

            loss = loss
            loss_array = loss.data.cpu().numpy()
            loss_sum = torch.sum(loss, 1)
            lengths = input_unpacked[1].to(device)
            lengths_array = lengths.data.cpu().numpy()
            loss_mean = loss_sum[:, 0] / lengths.float()
            loss = torch.sum(loss_mean)

            loss.backward()
            optimizer.step()

            print_loss_total += loss
            if iteration % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print(print_loss_avg)

            del x_input
            del x_target
            del out
            del output_unpacked
            del lengths
            del dec_hidden
            gc.collect()

        print('----epoch ' + str(epoch) + ' test----')
        iteration = 0
        print_loss_total_test = 0
        for i_batch, sample_batched in enumerate(test_loader):
            iteration += 1
            x_input = input_packing(sample_batched).to(device)

            hidden = encoder_model(x_input).to(device)
            conn = fully_connect(hidden).contiguous().to(device)

            out_list = []
            loss_mean_test = torch.zeros(batch_size, device=device)
            for bat in range(len(sample_batched)):
                dec_out = [torch.zeros(1, 1, 1, device=device)]
                out_vector = torch.zeros(sample_batched[bat].shape[0], 1, device=device)
                dec_hidden = conn[:, bat:bat + 1, :].contiguous()
                for word in range(out_vector.shape[0]):
                    dec_out = input_packing(dec_out).to(device)
                    out, dec_hidden = decoder_model(dec_out, dec_hidden)
                    dec_hidden = dec_hidden.contiguous()
                    dec_out = out.contiguous()
                    out_vector[word] = dec_out
                out_list.append(out_vector)
                loss_mean_test[bat] = criterion_val(out_vector, sample_batched[bat])
            loss_mean_test_array = loss_mean_test.cpu().detach().numpy()
            loss_test = torch.sum(loss_mean_test)

            print_loss_total_test += loss_test
            if iteration % print_every_val == 0:
                print_loss_avg_test = print_loss_total_test / print_every_val
                print_loss_total_test = 0
                print(print_loss_avg_test)

            del out_list
            gc.collect()

        torch.save(encoder_model.state_dict(), '../model_parameters/encoder_model_' + str(num_traces) + '.pt')
        torch.save(decoder_model.state_dict(), '../model_parameters/decoder_model_' + str(num_traces) + '.pt')
        torch.save(fully_connect.state_dict(), '../model_parameters/fully_connected_' + str(num_traces) +
                   '.pt')
