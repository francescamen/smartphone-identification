
"""
    confusion_matrix_analysis: computes the confusion matrices

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

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    confusion_matrix = np.load('../outputs/confusion_matrix_test.npy')

    number_users = confusion_matrix.shape[0]

    precision = np.zeros((number_users, 1))
    recall = np.zeros((number_users, 1))
    f_score = np.zeros((number_users, 1))
    true_positive = np.diag(confusion_matrix)
    false_positive = np.sum(confusion_matrix, axis=0) - true_positive
    false_negative = np.sum(confusion_matrix, axis=1) - true_positive
    for user in range(1, number_users + 1):
        if true_positive[user - 1] > 0:
            precision[user - 1] = true_positive[user - 1] / (true_positive[user - 1] + false_positive[user - 1])
            recall[user - 1] = true_positive[user - 1] / (true_positive[user - 1] + false_negative[user - 1])
            f_score[user - 1] = 2 * precision[user - 1] * recall[user - 1] / (precision[user - 1] + recall[user - 1])
        else:
            precision[user - 1] = 0
            recall[user - 1] = 0
            f_score[user - 1] = 0

    f_score_mean = np.mean(f_score)

    confusion_matrix_normaliz_row = confusion_matrix / np.sum(confusion_matrix, axis=1).reshape(-1, 1)
    confusion_matrix_normaliz_column = \
        confusion_matrix_normaliz_row / np.sum(confusion_matrix_normaliz_row, axis=0).reshape(1, -1)
    max_columns = np.amax(confusion_matrix_normaliz_column, axis=0)
    sum_max_columns = np.sum(max_columns)

    correct_windows = np.sum(np.diag(confusion_matrix))
    number_windows = np.sum(np.sum(confusion_matrix, 1))
    perc_correct_window = correct_windows/number_windows * 100
    print('perc_correct_windows: ' + str(perc_correct_window))

    confusion_matrix_majority = np.zeros(confusion_matrix_normaliz_row.shape)
    correct = 0
    for r in range(0, confusion_matrix_normaliz_row.shape[0]):
        index_max = np.argmax(confusion_matrix_normaliz_row[r, :])
        val_max = np.max(confusion_matrix_normaliz_row[r, :])
        indices_maxs = [i for i, j in enumerate(confusion_matrix_normaliz_row[r, :]) if j == val_max]
        confusion_matrix_majority[r, indices_maxs] = 1/len(indices_maxs)
        if index_max == r and len(indices_maxs) == 1:
            correct = correct + 1
    perc_correct_user = correct/number_users * 100
    print('perc_correct_users: ' + str(perc_correct_user))

    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 5)
    ax = axes.flat

    im1 = ax[0].pcolor(np.linspace(0.5, number_users + 0.5, number_users + 1),
                       np.linspace(0.5, number_users + 0.5, number_users + 1),
                       np.transpose(confusion_matrix_normaliz_row),
                       cmap='Blues', edgecolors='black', vmin=0, vmax=1)
    ax[0].set_title(r"$\bf{" + "Normalized" + "}$" + " " r"$\bf{" + "confusion" + "}$" + " " r"$\bf{" + "matrix" + "}$",
                    FontSize=14)
    ax[0].set_xlabel('Actual user', FontSize=14)
    ax[0].set_xticks(np.linspace(1, number_users, number_users), minor=True)
    ax[0].set_yticks(np.linspace(1, number_users, number_users), minor=True)
    ax[0].set_ylabel('Predicted user', FontSize=14)
    ax[0].tick_params(axis="x", labelsize=11)
    ax[0].tick_params(axis="y", labelsize=11)

    im2 = ax[1].pcolor(np.linspace(0.5, number_users + 0.5, number_users + 1),
                       np.linspace(0.5, number_users + 0.5, number_users + 1), np.transpose(confusion_matrix_majority),
                       cmap='Blues', edgecolors='black', vmin=0, vmax=1)

    ax[1].set_title(r"$\bf{" + "Majority" + "}$" + " " r"$\bf{" + "score" + "}$", FontSize=14)
    ax[1].set_xlabel('Actual user', FontSize=14)
    ax[1].set_xticks(np.linspace(1, number_users, number_users), minor=True)
    ax[1].set_yticks(np.linspace(1, number_users, number_users), minor=True)
    ax[1].set_ylabel('Predicted user', FontSize=14)
    ax[1].tick_params(axis="x", labelsize=11)
    ax[1].tick_params(axis="y", labelsize=11)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.ax.set_ylabel('Accuracy', FontSize=14)
    cbar.ax.tick_params(axis="y", labelsize=11)

    plt.savefig('../plots/cm_total.eps')

    plt.show()
