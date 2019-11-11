
"""
    Battacharyya_distance: computes distance metrics

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
import math as mt
import pickle
import matplotlib.gridspec as gridspec
import scipy.io as sio
matplotlib.use('TkAgg')
plt.switch_backend('agg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('num_clusters', help='Number of classes for clustering', type=int)
    args = parser.parse_args()

    num_traces = 40

    num_clusters = args.num_clusters
    compute_pdf = True

    if compute_pdf:
        # CLUSTERING
        with open('../processed_files/gauss.txt', "rb") as fp:  # Unpickling
            gauss = pickle.load(fp)

        # COMPUTE THE USERS PDF ON THE TRAINING DATA
        winLen = 3000
        winStep = 30
        users_pdf = np.zeros((num_traces, num_clusters))

        for idx in range(num_traces):
            with open('../processed_files/hiddens_train_user' + str(idx) + '.txt',
                      "rb") as fp:  # Unpickling
                sentences = pickle.load(fp)
            with open('../processed_files/times_train_user' + str(idx) + '.txt',
                      "rb") as fp:  # Unpickling
                time = pickle.load(fp)

            time[:, 1] = time[:, 1] - time[0, 1]
            end = time[-1, 1]
            num_groups = int(mt.ceil((end - winLen) / winStep))
            frequency_vector = np.zeros((num_groups, num_clusters))
            for i in range(num_groups):
                new_start = min(list(time[:, 1]), key=lambda x: abs(x - i * winStep))
                new_end = min(list(time[:, 1]), key=lambda x: abs(x - (new_start + winLen)))
                new_start_idx = np.argwhere(time[:, 1] == new_start)[0, 0]
                new_end_idx = np.argwhere(time[:, 1] == new_end)[0, 0]

                sentences_sampled_user = sentences[new_start_idx:new_end_idx, :]
                time_sampled = time[new_start_idx:new_end_idx, :]

                hidden_batched_vector = sentences_sampled_user.cpu().data.numpy()
                hidden_batched_vector = hidden_batched_vector
                labels_tot = gauss.predict(hidden_batched_vector)
                histog = np.histogram(labels_tot, bins=np.linspace(0, num_clusters, num_clusters + 1))
                freq = histog[0]
                frequency_vector[i, :] = freq / np.amax(freq)
            frequency_vector_mean = np.mean(frequency_vector, axis=0)
            users_pdf[idx, :] = frequency_vector_mean / np.sum(frequency_vector_mean)

        with open('../outputs/users_pdf.txt', "wb") as fp:  # Pickling
            pickle.dump(users_pdf, fp)

        # COMPUTE THE USERS PDF ON THE TEST DATA
        winLen = 3000
        winStep = 30
        users_pdf = np.zeros((num_traces, num_clusters))

        for idx in range(num_traces):
            with open('../processed_files/hiddens_test_user' + str(idx) + '.txt',
                      "rb") as fp:  # Unpickling
                sentences = pickle.load(fp)
            with open('../processed_files/times_test_user' + str(idx) + '.txt',
                      "rb") as fp:  # Unpickling
                time = pickle.load(fp)

            time[:, 1] = time[:, 1] - time[0, 1]
            end = time[-1, 1]
            num_groups = int(mt.ceil((end - winLen) / winStep))
            frequency_vector = np.zeros((num_groups, num_clusters))
            for i in range(num_groups):
                new_start = min(list(time[:, 1]), key=lambda x: abs(x - i * winStep))
                new_end = min(list(time[:, 1]), key=lambda x: abs(x - (new_start + winLen)))
                new_start_idx = np.argwhere(time[:, 1] == new_start)[0, 0]
                new_end_idx = np.argwhere(time[:, 1] == new_end)[0, 0]

                sentences_sampled_user = sentences[new_start_idx:new_end_idx, :]
                time_sampled = time[new_start_idx:new_end_idx, :]

                hidden_batched_vector = sentences_sampled_user.cpu().data.numpy()
                hidden_batched_vector = hidden_batched_vector
                labels_tot = gauss.predict(hidden_batched_vector)
                histog = np.histogram(labels_tot, bins=np.linspace(0, num_clusters, num_clusters + 1))
                freq = histog[0]
                frequency_vector[i, :] = freq / np.amax(freq)
            frequency_vector_mean = np.mean(frequency_vector, axis=0)
            users_pdf[idx, :] = frequency_vector_mean / np.sum(frequency_vector_mean)

        with open('../outputs/users_pdf_test.txt', "wb") as fp:  # Pickling
            pickle.dump(users_pdf, fp)

    # COMPUTE THE DISTANCE METRICS FOR THE TRAIN DATA
    with open('../outputs/users_pdf.txt', "rb") as fp:  # Unpickling
        users_pdf = pickle.load(fp)

    BC = np.zeros((num_traces, num_traces))
    for idx1 in range(num_traces):
        pdf_user1 = users_pdf[idx1, :]
        for idx2 in range(num_traces):
            if idx2 > idx1:
                pdf_user2 = users_pdf[idx2, :]
                BC[idx1, idx2] = -mt.log(np.sum(np.sqrt(np.multiply(pdf_user1, pdf_user2))))

    # COMPUTE THE DISTANCE METRICS FOR THE TEST DATA
    with open('../outputs/users_pdf_test.txt', "rb") as fp:  # Unpickling
        users_pdf_test = pickle.load(fp)

    BC_test = np.zeros((num_traces, num_traces))
    for idx1 in range(num_traces):
        pdf_user1 = users_pdf_test[idx1, :]
        for idx2 in range(num_traces):
            if idx2 > idx1:
                pdf_user2 = users_pdf_test[idx2, :]
                BC_test[idx1, idx2] = -mt.log(np.sum(np.sqrt(np.multiply(pdf_user1, pdf_user2))))

    with open('../outputs/traces_test_accuracy.txt', "rb") as fp:  # Unpickling
        traces_test_accuracy = pickle.load(fp)

    BC_test_line = BC_test.reshape(-1)
    traces_test_accuracy_line = traces_test_accuracy.reshape(-1)

    mApp_structure = sio.loadmat('../input_files/mApp.mat')
    mApp = mApp_structure['mApp']

    hamming_dist_app = np.zeros((num_traces, num_traces))
    matrix_app = np.zeros((num_traces, num_traces), dtype=int)
    for idx1 in range(num_traces):
        app_user1 = mApp[idx1, :]
        for idx2 in range(num_traces):
            app_user2 = mApp[idx2, :]
            if idx2 > idx1:
                dist_coeff = np.sum(np.abs(app_user1 - app_user2)) / (mt.pow(app_user1.shape[0], 2))
                hamming_dist_app[idx1, idx2] = dist_coeff
            matrix_app[idx1, idx2] = np.sum(app_user1 * app_user2)  # app in common

    hamming_dist_app_line = hamming_dist_app.reshape(-1)

    fig = plt.figure(1)
    gs = gridspec.GridSpec(3, 3, wspace=0.38, hspace=0.68)
    ax1 = plt.subplot(gs[:, :-1])
    ax2 = plt.subplot(gs[0, -1:])
    ax3 = plt.subplot(gs[1, -1:])
    ax4 = plt.subplot(gs[2, -1:])
    ax1.scatter(hamming_dist_app_line / np.amax(hamming_dist_app_line), traces_test_accuracy_line, s=10, linewidths=0,
                c='firebrick', marker='x', label='Actual applications', linewidth=0.7)
    ax1.scatter(BC_test_line, traces_test_accuracy_line, s=10, linewidths=0, edgecolors='midnightblue', marker='o',
                label='Estimated applications', linewidth=0.4, facecolors='none')
    ax1.grid()
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_xlabel('Distance', FontSize=16)
    ax1.set_ylabel('Successful user disambiguation rate', FontSize=16)
    ax1.legend()

    idx = 3
    ax2.bar(np.linspace(1, num_clusters, num_clusters), users_pdf[idx, :], color='midnightblue')
    ax2.grid()
    ax2.set_title('User' + str(idx + 1), FontSize=16)
    ax2.set_xlabel('Clusters', FontSize=12)
    ax2.set_ylabel('Probability', FontSize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_xticks(np.linspace(1, num_clusters, num_clusters))
    ax2.locator_params(axis='x', nbins=5)
    ax2.locator_params(axis='y', nbins=5)

    idx = 4
    ax3.bar(np.linspace(1, num_clusters, num_clusters), users_pdf[idx, :], color='midnightblue')
    ax3.grid()
    ax3.set_title('User' + str(idx + 1), FontSize=16)
    ax3.set_xlabel('Clusters', FontSize=12)
    ax3.set_ylabel('Probability', FontSize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax3.set_xticks(np.linspace(1, num_clusters, num_clusters))
    ax3.locator_params(axis='x', nbins=5)
    ax3.locator_params(axis='y', nbins=5)

    idx = 11
    ax4.bar(np.linspace(1, num_clusters, num_clusters), users_pdf[idx, :], color='midnightblue')
    ax4.grid()
    ax4.set_title('User' + str(idx + 1), FontSize=16)
    ax4.set_xlabel('Clusters', FontSize=12)
    ax4.set_ylabel('Probability', FontSize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax4.set_xticks(np.linspace(1, num_clusters, num_clusters))
    ax4.locator_params(axis='x', nbins=5)
    ax4.locator_params(axis='y', nbins=5)

    fig.tight_layout()
    fig.set_size_inches(w=11, h=7)
    fig.savefig('../plots/batt_pdf.eps')

    # Confusion Matrix with numbers of common applications
    confusion_matrix = np.load('../outputs/confusion_matrix_test.npy')

    number_users = confusion_matrix.shape[0]

    confusion_matrix_normaliz_row = confusion_matrix / np.sum(confusion_matrix, axis=1).reshape(-1, 1)
    confusion_matrix_normaliz_column = \
        confusion_matrix_normaliz_row / np.sum(confusion_matrix_normaliz_row, axis=0).reshape(1, -1)
    max_columns = np.amax(confusion_matrix_normaliz_column, axis=0)
    sum_max_columns = np.sum(max_columns)

    correct_windows = np.sum(np.diag(confusion_matrix))
    number_windows = np.sum(np.sum(confusion_matrix, 1))
    perc_correct_window = correct_windows / number_windows * 100
    print('perc_correct_window: ' + str(perc_correct_window))

    fig = plt.figure(6)
    ax = plt.axes()
    fig.set_size_inches(6, 5)
    max_matrix_app = np.amax(confusion_matrix_normaliz_row)
    im1 = plt.pcolor(np.linspace(0.5, num_traces + 0.5, num_traces + 1),
                     np.linspace(0.5, num_traces + 0.5, num_traces + 1),
                     np.transpose(confusion_matrix_normaliz_row),
                     cmap='Blues', edgecolors='black', vmin=0, vmax=max_matrix_app)
    ax.set_title(r"$\bf{" + "Normalized" + "}$" + " " r"$\bf{" + "confusion" + "}$" + " " r"$\bf{" + "matrix" + "}$",
                 FontSize=9)
    ax.set_xlabel('Actual user', FontSize=8)

    ax.set_xticks(np.linspace(1, num_traces, num_traces), minor=True)
    ax.set_yticks(np.linspace(1, num_traces, num_traces), minor=True)
    ax.set_ylabel('Predicted user', FontSize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    for y_ax in range(matrix_app.shape[0]):
        for x_ax in range(matrix_app.shape[1]):
            col = 'k'
            if confusion_matrix_normaliz_row[x_ax, y_ax] > 0.6:  # [x, y] because plot the transpose version
                col = 'w'
            ax.text(x_ax + 1, y_ax + 1, '%d' % matrix_app[y_ax, x_ax], horizontalalignment='center',
                    verticalalignment='center', fontsize=4, color=col)

    cbar = fig.colorbar(im1)
    cbar.ax.set_ylabel('Accuracy', FontSize=8)
    cbar.ax.tick_params(axis="y", labelsize=7)

    fig.savefig('../plots/users_apps.eps')

    with open('../outputs/matrix_app.txt', "wb") as fp:  # Pickling
        pickle.dump(matrix_app, fp)
