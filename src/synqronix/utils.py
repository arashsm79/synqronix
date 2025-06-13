import scipy.io
import numpy as np

def print_keys(d, indent=0):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            print('  ' * indent + str(key) + ': ' + str(value.shape))
        else:
            print('  ' * indent + str(key))
        if isinstance(value, dict):
            print_keys(value, indent + 2)


def process_mat(mat_data):
    def mat_cell_to_dict(mt):
        clean_data = {}
        keys = mt.dtype.names
        for key_idx, key in enumerate(keys):
            clean_data[key] = np.squeeze(mt[key_idx]) if isinstance(mt[key_idx], np.ndarray) else mt[key_idx]
        return clean_data

    def planewise_mat_cell_to_dict(mt):
        clean_data = {}
        for plane_id in range(len(mt[0])):
            keys = mt[0, plane_id].dtype.names
            clean_data[plane_id] = {}
            for key_idx, key in enumerate(keys):
                clean_data[plane_id][key] = np.squeeze(mt[0, plane_id][key_idx]) if isinstance(mt[0, plane_id][key_idx], np.ndarray) else mt[0, plane_id][key_idx]
        return clean_data

    mt = {}
    mt['BFInfo']   = planewise_mat_cell_to_dict(mat_data['BFinfo'])
    mt['CellInfo'] = planewise_mat_cell_to_dict(mat_data['CellInfo'])
    mt['CorrInfo'] = planewise_mat_cell_to_dict(mat_data['CorrInfo'])
    mt['allZCorrInfo'] = mat_cell_to_dict(mat_data['allZCorrInfo'][0, 0])

    for cord_key in ['allxc', 'allyc', 'allzc', 'zDFF']:
        mt[cord_key] = {}
        for p in range(mat_data[cord_key].shape[0]):
            mt[cord_key][p] = mat_data[cord_key][p, 0]

    mt['exptVars'] = mat_cell_to_dict(mat_data['exptVars'][0, 0])
    mt['selectZCorrInfo'] = mat_cell_to_dict(mat_data['selectZCorrInfo'][0, 0])
    mt['stimInfo'] = planewise_mat_cell_to_dict(mat_data['stimInfo'])
    mt['zStuff'] = planewise_mat_cell_to_dict(mat_data['zStuff'])

    return mt