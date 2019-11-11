
"""
    sentences_clustering: clusters the codes into classes using the GMM-based clustering algorithm

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
from sklearn.mixture import BayesianGaussianMixture
import pickle
plt.switch_backend('agg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('num_clusters', help='Number of classes for clustering', type=int)
    args = parser.parse_args()

    num_traces = 40

    # ---------------------------------------------------------------------------------------------------------------- #
    # SENTENCES CLUSTERING #

    hidden_sentences = []
    lengths = []
    for idx in range(num_traces):
        with open('../processed_files/hiddens_train_user' + str(idx) + '.txt',
                  "rb") as fp:  # Unpickling
            sentences_train = pickle.load(fp)
        hidden_sentences.extend(sentences_train.cpu().data.numpy())
        lengths.append(len(sentences_train))

    hidden_sentences = np.asarray(hidden_sentences)
    lengths = np.asarray(lengths)

    sentences = hidden_sentences

    num_clusters = args.num_clusters
    gauss = BayesianGaussianMixture(n_components=num_clusters, covariance_type='diag', max_iter=200).fit(
        sentences)

    with open('../processed_files/gauss.txt', "wb") as fp:  # Pickling
        pickle.dump(gauss, fp)
