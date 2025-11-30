import h5py
from pprint import pprint
import ast
import glob
import os
import numpy as np
from matplotlib import pyplot as plt


def list_hdf5_files(folder_path):
    # Search for all .h5 or .hdf5 files in the folder
    hdf5_files = glob.glob(os.path.join(folder_path, '*.h5')) + glob.glob(os.path.join(folder_path, '*.hdf5'))
    return hdf5_files


def get_data_from_file(file_path, dim):
    # Open an HDF5 file
    with h5py.File(file_path, 'r') as file:
        # Access the 'Data' group and then the 'Data' dataset
        data_group = file['Data']
        data_dataset = data_group['Data']

        # Print dataset information
        # print("Dataset 'Data' info:")
        # print("Shape:", data_dataset.shape)
        # print("Datatype:", data_dataset.dtype)

        data = data_dataset[()]

        data = np.array(data)
        if dim == 1:
            x = np.array(data).T[0][0]
            y = np.array(data).T[0][1]
            data = [x, y]
            return data, None, None

        elif dim == 2:
            x = data.T[0][0][:]
            y = data[0][1][:]
            matrix = []
            for i in range(0, len(data.T)):
                matrix.append(data.T[i][2][:])

            data = [x, y, matrix]

            commends = file.attrs['comment']
            dict_obj = ast.literal_eval(commends)
            args = dict_obj['args']
            exp_args = dict_obj['exp_args']
            return data, args, exp_args


if __name__ == "__main__":
    file_path = '/Users/asafsolonnikov/Documents/GitHub/Power-Broadening-2/graphs_for_poster/lorentzian/data/T1-qubit-spectroscopy__9.hdf5'
    get_data_from_file(file_path, dim=2)
