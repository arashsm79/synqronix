import scipy.io
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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


def create_neuron_info_df(mat_data: dict, save_path):
    """
    Create a comprehensive dataframe with neuron information from layers 1-6.
    
    Args:
        mat_data: Processed MATLAB data dictionary
        save_path: Path to save the resulting dataframe
    
    Returns:
        pandas.DataFrame: Comprehensive neuron information dataframe
    """
    neuron_rows = []
    global_idx = 0
    
    # Collect all zDFF data for PCA
    all_zdff = []
    for layer in range(1, 6):  # layers 1-5
        all_zdff.append(mat_data['zDFF'][layer])
    
    # Concatenate all zDFF data and compute PCA
    all_zdff_concat = np.vstack(all_zdff)
    pca = PCA(n_components=4)
    all_pcs = pca.fit_transform(all_zdff_concat)
    pc_start_idx = 0
    
    # Process each layer (1-5)
    for layer in range(1, 6):
        num_neurons = mat_data['zDFF'][layer].shape[0]
        
        for neuron_idx in range(num_neurons):
            neuron_info = {
                'global_idx': global_idx,
                'layer': layer,
                'neuron_idx_in_layer': neuron_idx,
                'x': mat_data['allxc'][layer][neuron_idx, 0],
                'y': mat_data['allyc'][layer][neuron_idx, 0],
                'z': mat_data['allzc'][layer][neuron_idx, 0],
                'activity': mat_data['zDFF'][layer][neuron_idx, :].tolist(),
            }
            
            # Add per-trial activity (trialDFF from zStuff)
            trial_dff = mat_data['zStuff'][layer]['trialDFF']  # Shape: (9, 10, 57, 23)
            neuron_info['per_trial_activity'] = trial_dff[:, :, neuron_idx, :].tolist()
            
            # Add BF information
            bf_info = mat_data['BFInfo'][layer]
            neuron_info['BFval'] = bf_info['BFval'][neuron_idx]
            neuron_info['BFresp'] = bf_info['BFresp'][neuron_idx]
            
            # Add PCA components
            pc_idx = pc_start_idx + neuron_idx
            neuron_info['PC'] = all_pcs[pc_idx, :].tolist()
            
            # Add correlations with all other neurons using selectZCorrInfo
            sig_corrs = mat_data['selectZCorrInfo']['SigCorrs']
            # Map global index to selectZCorrInfo index (layers 1-5)
            neuron_info['global_corr'] = sig_corrs[global_idx, :].tolist()
            
            # Add distances to other neurons in the same layer
            cell_dists = mat_data['CellInfo'][layer]['cellDists']
            neuron_info['layer_distances'] = cell_dists[neuron_idx, :].tolist()
            
            neuron_rows.append(neuron_info)
            global_idx += 1
        
        # Update PC start index for next layer
        pc_start_idx += num_neurons
    
    # Create DataFrame
    df = pd.DataFrame(neuron_rows)
    
    # Save to file
    df.to_pickle(save_path)
    
    return df
