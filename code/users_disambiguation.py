
"""
    users_disambiguation: computes the accuracy in the separation between two users

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
import tensorflow as tf
from tensorflow.python.framework import ops
import math as mt
import pickle
import gc
from pathlib import Path
plt.switch_backend('agg')


def convert_to_one_hot(y, c):
    y = np.eye(c)[(y - 1).reshape(-1)].T
    return y


def create_placeholders(max_length_sequences, n_y):
    x = tf.placeholder(tf.float32, shape=(None, max_length_sequences, 1))
    y = tf.placeholder(tf.float32, shape=(None, n_y))
    return x, y


def initialize_parameters():
    tf.set_random_seed(1)
    w1 = tf.get_variable("W1", [10, 1, 5], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    w2 = tf.get_variable("W2", [5, 5, 10], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    w3 = tf.get_variable("W3", [5, 10, 20], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    w4 = tf.get_variable("W4", [3, 20, 30], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"w1": w1, "w2": w2, "w3": w3, "w4": w4}
    return parameters


def cnn_encoder(x, parameters):
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    w4 = parameters['w4']

    z1 = tf.nn.conv1d(x, w1, stride=1, padding='SAME')
    a1 = tf.nn.relu(z1)
    d1 = tf.nn.dropout(a1, 0.8)

    z2 = tf.nn.conv1d(d1, w2, stride=1, padding='SAME')
    a2 = tf.nn.relu(z2)
    d2 = tf.nn.dropout(a2, 0.8)

    z3 = tf.nn.conv1d(d2, w3, stride=1, padding='SAME')
    a3 = tf.nn.relu(z3)
    d3 = tf.nn.dropout(a3, 0.8)

    z4 = tf.nn.conv1d(d3, w4, stride=1, padding='SAME')
    a4 = tf.nn.relu(z4)

    p6 = tf.contrib.layers.flatten(a4)
    p6 = tf.layers.dropout(p6, rate=0.2)

    z6 = tf.contrib.layers.fully_connected(p6, num_selected_traces, activation_fn=None)
    return z6


def compute_cost(z, y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y))
    return cost


def confusion_matrix_computation(num_users, test_labels, prediction_test_labels):
    confusion_matrix = np.zeros((num_users, num_users))
    for i_act in range(1, num_users + 1):  # actual user
        indices_act_i = np.argwhere(test_labels == i_act)[:, 0]
        for j_act in range(1, num_users + 1):  # predicted user
            indices_act_j = np.argwhere(prediction_test_labels == j_act)[:, 0]
            intersect = set(indices_act_i).intersection(indices_act_j)
            confusion_matrix[i_act - 1, j_act - 1] = len(intersect)
    return confusion_matrix


def random_mini_batches(x, y, mini_batch_size=64, seed=0):
    m = x.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_x = x[permutation, :, :]
    shuffled_y = y[permutation, :]

    num_complete_minibatches = mt.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitioning
    for k in range(0, num_complete_minibatches):
        mini_batch_x = shuffled_x[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch_y = shuffled_y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_x[num_complete_minibatches * mini_batch_size: m, :, :]
        mini_batch_y = shuffled_y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches


def model(x_train, y_train, x_test, y_test, test_accuracy_old, num_epochs=100, minibatch_size=64, print_cost=True):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)
    seed = 3
    (m, max_length_sequences, depth) = x_train.shape  # m is the number of sequences
    (m_y, n_y) = y_train.shape

    costs = []

    x, y = create_placeholders(max_length_sequences, n_y)

    parameters = initialize_parameters()

    z6 = cnn_encoder(x, parameters)

    cost = compute_cost(z6, y)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    init = tf.global_variables_initializer()

    train_accuracy = 0
    test_accuracy = 0

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(x_train, y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_x, minibatch_y) = minibatch

                _, temp_cost = sess.run([optimizer, cost], feed_dict={x: minibatch_x, y: minibatch_y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost
            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))

                predict_op = tf.argmax(z6, 1)
                correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))

                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

                train_accuracy = accuracy.eval({x: x_train, y: y_train})

                test_accuracy = accuracy.eval({x: x_test, y: y_test})

                print("Train accuracy: %f" % train_accuracy)
                print("Test accuracy:%f " % test_accuracy)

            if print_cost and epoch % 10 == 0:
                costs.append(minibatch_cost)

            # Save the prediction
            prediction_train = sess.run(z6, {x: x_train, y: y_train})
            prediction_test = sess.run(z6, {x: x_test, y: y_test})

            if test_accuracy > test_accuracy_old:
                test_accuracy_old = test_accuracy

        gc.collect()

        sess.close()

    return train_accuracy, test_accuracy, parameters, prediction_train, prediction_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('num_clusters', help='Number of classes for clustering', type=int)
    parser.add_argument('epochs', help='Number of epochs', type=int)
    args = parser.parse_args()

    num_traces = 40

    # ---------------------------------------------------------------------------------------------------------------- #
    # CONVOLUTIONAL NETWORK USER IDENTIFICATION #

    num_clusters = args.num_clusters
    with open('../processed_files/gauss.txt', "rb") as fp:  # Unpickling
        gauss = pickle.load(fp)

    num_selected_traces = 2

    num_epcs = args.epochs

    my_file = Path('../outputs/traces_test_accuracy.txt')
    if my_file.is_file():
        with open('../outputs/traces_test_accuracy.txt', "rb") as fp:  # Unpickling
            traces_test_accuracy = pickle.load(fp)
        start_idx1 = np.argwhere(np.diag(traces_test_accuracy, 1) == 0)[0, 0] - 1
        start_idx2 = np.argwhere(traces_test_accuracy[start_idx1, start_idx1 + 1:] == 0)
        if len(start_idx2) == 0:
            start_idx1 = start_idx1 + 1
            start_idx2 = start_idx1 + 1
        else:
            start_idx2 = start_idx2[0, 0] + start_idx1 + 1
    else:
        traces_test_accuracy = np.zeros((num_traces, num_traces))
        start_idx1 = 0
        start_idx2 = 1

    for idx1 in range(start_idx1, num_traces):
        if start_idx1 != idx1:
            start_idx2 = idx1 + 1
        for idx2 in range(start_idx2, num_traces):
            print(' ')
            print('idx1 ' + str(idx1))
            print('idx2 ' + str(idx2))
            selected_indices = np.asarray([idx1, idx2])

            len_win = 3000  # in seconds
            step_win = 30  # in seconds
            users_sliding = []
            index_vector = []
            user_new = 0
            for user in selected_indices:
                print(user)
                with open('../processed_files/hiddens_train_user' + str(user) + '.txt',
                          "rb") as fp:  # Unpickling
                    sentences_u = pickle.load(fp)
                with open('../processed_files/times_train_user' + str(user) + '.txt',
                          "rb") as fp:  # Unpickling
                    time_u = pickle.load(fp)
                sentences_u = sentences_u.data.cpu().numpy()
                time_ass = time_u[:, 1] - time_u[0, 1]
                length_user = time_ass[-1]
                sentences_k = []
                num_window = mt.ceil((length_user - len_win) / step_win)

                print('user ' + str(user) + ': train window ' + str(num_window))
                for t in range(num_window):
                    indices = np.argwhere((time_ass > t * step_win) & (time_ass < t * step_win + len_win))[:, 0]
                    if indices.shape[0] > 1:
                        sentences_k.append(sentences_u[indices[0]:indices[-1], :])

                frequency_vector = np.zeros((len(sentences_k), num_clusters))
                for s in range(len(sentences_k)):
                    sentence = np.asarray(sentences_k[s])
                    hidden_batched_vector = sentence
                    labels_tot = gauss.predict(hidden_batched_vector)
                    histog = np.histogram(labels_tot, bins=np.linspace(0, num_clusters, num_clusters + 1))
                    freq = histog[0]
                    frequency_vector[s, :] = freq / np.amax(freq)

                users_sliding.append(frequency_vector)
                index_vector.extend(len(sentences_k) * [user_new])
                user_new = user_new + 1

            input_matrix_tr = np.vstack(users_sliding)
            train_label = np.asarray(index_vector) + 1

            users_sliding = []
            index_vector = []
            len_win = 3000  # in seconds
            step_win = 30  # in seconds
            user_new = 0
            for user in selected_indices:
                print(user)
                with open('../processed_files/hiddens_test_user' + str(user) + '.txt',
                          "rb") as fp:  # Unpickling
                    sentences_u = pickle.load(fp)
                with open('../processed_files/times_test_user' + str(user) + '.txt',
                          "rb") as fp:  # Unpickling
                    time_u = pickle.load(fp)
                sentences_u = sentences_u.data.cpu().numpy()
                time_ass = time_u[:, 1] - time_u[0, 1]
                length_user = time_ass[-1]
                sentences_k = []
                num_window = mt.ceil((length_user - len_win) / step_win)

                print('user ' + str(user) + ': test window ' + str(num_window))
                for t in range(num_window):
                    indices = np.argwhere((time_ass > t * step_win) & (time_ass < t * step_win + len_win))[:, 0]
                    if indices.shape[0] > 1:
                        sentences_k.append(sentences_u[indices[0]:indices[-1], :])

                frequency_vector = np.zeros((len(sentences_k), num_clusters))
                for s in range(len(sentences_k)):
                    sentence = np.asarray(sentences_k[s])
                    hidden_batched_vector = sentence
                    labels_tot = gauss.predict(hidden_batched_vector)
                    histog = np.histogram(labels_tot, bins=np.linspace(0, num_clusters, num_clusters + 1))
                    freq = histog[0]
                    frequency_vector[s, :] = freq/np.amax(freq)

                users_sliding.append(frequency_vector)
                index_vector.extend(len(sentences_k) * [user_new])
                user_new = user_new + 1

            input_matrix_test = np.vstack(users_sliding)
            test_label = np.asarray(index_vector) + 1

            input_matrix_tr = input_matrix_tr.reshape((input_matrix_tr.shape[0], input_matrix_tr.shape[1], 1))
            input_matrix_test = input_matrix_test.reshape((input_matrix_test.shape[0], input_matrix_test.shape[1], 1))

            output_matrix_tr = convert_to_one_hot(train_label, num_selected_traces).T
            output_matrix_test = convert_to_one_hot(test_label, num_selected_traces).T

            test_acc_old = 0
            _, test_accuracy_new, _, _, _ = model(input_matrix_tr, output_matrix_tr, input_matrix_test,
                                                  output_matrix_test, test_acc_old, num_epochs=num_epcs)

            traces_test_accuracy[idx1, idx2] = test_accuracy_new

            with open('../processed_files/traces_test_accuracy.txt', "wb") as fp:  # Pickling
                pickle.dump(traces_test_accuracy, fp)
            del input_matrix_tr
            del input_matrix_test
            del output_matrix_test
            del output_matrix_tr
            del test_label
            gc.collect()
