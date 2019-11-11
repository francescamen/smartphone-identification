
"""
    data_loading_preprocessing: loads the Matlab data and outputs the files to be used for further processing

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
import scipy.io as sio
import pickle
plt.switch_backend('agg')


if __name__ == '__main__':
    num_traces = 40

    # ---------------------------------------------------------------------------------------------------------------- #
    # FILE LOADING #

    # Load the Matlab files
    mPack_structure = sio.loadmat('../input_files/mPack.mat')
    mPack = mPack_structure['mPack']

    mTime_structure = sio.loadmat('../input_files/mTime.mat')
    mTime = mTime_structure['mTime']

    mInfo_structure = sio.loadmat('../input_files/mInfo.mat')
    mInfo = mInfo_structure['mInfo']

    validInt_struct = sio.loadmat('../input_files/validInts.mat')
    validInts = validInt_struct['validInts']

    # Order the file with respect to the absolute time, ordering inside each trace
    traceNum = 1
    mTimeSelect = []
    mPackSelect = []
    mInfoSelect = []
    for trace in range(1, num_traces + 1):
        indices = np.argwhere(mInfo[:, -1] == trace)

        mTime_trace = mTime[indices[:, 0], :]
        mPack_trace = mPack[indices[:, 0], :]
        mInfo_trace = mInfo[indices[:, 0], :]

        mInfo_trace[:, -1] = traceNum  # to label each selected trace with increasing indices
        traceNum = traceNum + 1

        # concatenate all the information in a single vector in order to order all the information in the same manner
        array_to_be_ordered = np.zeros((mTime_trace.shape[0], mTime_trace.shape[1] + mPack_trace.shape[1] +
                                        mInfo_trace.shape[1]))
        array_to_be_ordered[:, 0:mTime_trace.shape[1]] = mTime_trace
        array_to_be_ordered[:, mTime_trace.shape[1]:mTime_trace.shape[1] + mPack_trace.shape[1]] = mPack_trace
        array_to_be_ordered[:, mTime_trace.shape[1] + mPack_trace.shape[1]:mTime_trace.shape[1] + mPack_trace.shape[1]
                                                    + mInfo_trace.shape[1]] = mInfo_trace

        # order the array
        array_ordered = array_to_be_ordered[array_to_be_ordered[:, 1].argsort()]

        # extract each singular information from the vector
        mTime_trace = array_ordered[:, 0:mTime_trace.shape[1]]
        mPack_trace = array_ordered[:, mTime_trace.shape[1]:mTime_trace.shape[1] + mPack_trace.shape[1]]
        mInfo_trace = array_ordered[:, mTime_trace.shape[1] + mPack_trace.shape[1]:mTime_trace.shape[1]
                                                            + mPack_trace.shape[1]
                                                            + mInfo_trace.shape[1]]

        # save the information in the matrices with all the traces
        mTimeSelect.append(mTime_trace)
        mPackSelect.append(mPack_trace)
        mInfoSelect.append(mInfo_trace)
    mTime = np.vstack(mTimeSelect)
    mPack = np.vstack(mPackSelect)
    mInfo = np.vstack(mInfoSelect)

    # Cancel the sentences with a length smaller than threshold, deleting also all the related information
    # to that aim, first concatenate all the information in a single vector
    array = np.zeros((mTime.shape[0], mTime.shape[1] + mPack.shape[1] + mInfo.shape[1]))
    array[:, 0:mTime.shape[1]] = mTime
    array[:, mTime.shape[1]:mTime.shape[1] + mPack.shape[1]] = mPack
    array[:, mTime.shape[1] + mPack.shape[1]:mTime.shape[1] + mPack.shape[1] + mInfo.shape[1]] = mInfo

    # create a new array in which to insert only the sentences with more than threshold samples
    new_array = np.zeros((mTime.shape[0], mTime.shape[1] + mPack.shape[1] + mInfo.shape[1]))
    index = 0
    threshold = 6  # the words smaller than that threshold will be discarded
    for i in range(array.shape[0]):
        if (array[i, mTime.shape[1] + 2 + threshold] != 0) or (np.sum(array[i, mTime.shape[1] + 2: mTime.shape[1] + 2
                                                                                              + threshold]) > 500):
            new_array[index, :] = array[i, :]
            index = index + 1
    new_array = new_array[:index, :]

    # Select only the patterns inside the valid intervals extracted with Matlab
    new_array_2 = np.zeros((new_array.shape[0], mTime.shape[1] + mPack.shape[1] + mInfo.shape[1]))
    index = 0
    for i in range(new_array.shape[0]):
        trace_index = int(new_array[i, mPack.shape[1] + mTime.shape[1] + mInfo.shape[1] - 1])
        start_valid_time = validInts[0][trace_index - 1][0, 0] + 5*60
        end_valid_time = validInts[0][trace_index - 1][0, 1]
        if end_valid_time > 37000:  # cut all traces at 10 hours
            end_valid_time = start_valid_time + 36030
        if (new_array[i, 1] > start_valid_time) and (new_array[i, 1] < end_valid_time):
            new_array_2[index, :] = new_array[i, :]
            index = index + 1
    new_array_2 = new_array_2[:index, :]

    mTime_new = new_array_2[:, 0:mTime.shape[1]]
    mPack_new = new_array_2[:, mTime.shape[1]:mTime.shape[1] + mPack.shape[1]]
    mInfo_new = new_array_2[:, mTime.shape[1] + mPack.shape[1]:mTime.shape[1] + mPack.shape[1] + mInfo.shape[1]]

    trainFraction = 0.8
    sentences_train_pad = []
    sentences_test_pad = []
    mInfo_train = []
    mInfo_test = []
    mTime_train = []
    mTime_test = []
    dir_train = []
    dir_test = []

    for trace in range(1, num_traces + 1):
        indices = np.argwhere(mInfo_new[:, -1] == trace)

        mPack_trace = mPack_new[indices[:, 0], :]
        mInfo_trace = mInfo_new[indices[:, 0], :]
        mTime_trace = mTime_new[indices[:, 0], :]

        time_ass = mTime_trace[:, 1] - mTime_trace[0, 1]
        length_user_time = time_ass[-1]
        trLen_time = int(length_user_time*trainFraction)

        trLen = np.argwhere(time_ass < trLen_time)[-1, 0]

        sentences_train_pad.extend(mPack_trace[:trLen, 2:]/20000)
        sentences_test_pad.extend(mPack_trace[trLen:, 2:]/20000)

        mInfo_train.extend(mInfo_trace[:trLen, :])
        mInfo_test.extend(mInfo_trace[trLen:, :])

        mTime_train.extend(mTime_trace[:trLen, :])
        mTime_test.extend(mTime_trace[trLen:, :])

        dir_train.extend(mPack_trace[:trLen, 0])
        dir_test.extend(mPack_trace[trLen:, 0])

    sentences_train = []
    sentences_test = []
    for i in range(len(sentences_train_pad)):
        idx = np.argwhere(sentences_train_pad[i] == 0)
        if idx.size != 0:
            sentences_train.append(sentences_train_pad[i][:idx[0, 0]])
        else:
            sentences_train.append(sentences_train_pad[i])

    for i in range(len(sentences_test_pad)):
        idx = np.argwhere(sentences_test_pad[i] == 0)
        if idx.size != 0:
            sentences_test.append(sentences_test_pad[i][:idx[0, 0]])
        else:
            sentences_test.append(sentences_test_pad[i])

    del sentences_train_pad
    del sentences_test_pad
    del mInfo
    del mTime
    del mPack
    del mInfo_new
    del mPack_new
    del mTime_new
    del array
    del new_array_2
    del new_array
    del validInts
    del validInt_struct
    del mPack_structure
    del mInfo_structure
    del mTime_structure

    with open('../processed_files/mInfo_train.txt', "wb") as fp:  # Pickling
        pickle.dump(mInfo_train, fp)
    with open('../processed_files/mInfo_test.txt', "wb") as fp:  # Pickling
        pickle.dump(mInfo_test, fp)

    with open('../processed_files/mTime_train.txt', "wb") as fp:  # Pickling
        pickle.dump(mTime_train, fp)
    with open('../processed_files/mTime_test.txt', "wb") as fp:  # Pickling
        pickle.dump(mTime_test, fp)

    with open('../processed_files/sentences_train.txt', "wb") as fp:  # Pickling
        pickle.dump(sentences_train, fp)
    with open('../processed_files/sentences_test.txt', "wb") as fp:  # Pickling
        pickle.dump(sentences_test, fp)
