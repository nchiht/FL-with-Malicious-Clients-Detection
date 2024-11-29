import os
from threading import Lock
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import torch
from flwr.common import NDArrays, Parameters, Scalar, parameters_to_ndarrays
# from natsort import natsorted

lock = Lock()
def flatten_params(params):
    """Transform a list of (layers-)parameters into a single vector of shape (n)."""
    return np.concatenate(params, axis=None).ravel()
def save_params(
    parameters, cid, params_dir="clients_params", remove_last=False, rrl=False
):
    """Save parameters in a file.

    Args:
    - parameters (ndarray): decoded parameters to append at the end of the file
    - cid (int): identifier of the client
    - remove_last (bool):
        if True, remove the last saved parameters and replace with "parameters"
    - rrl (bool):
        if True, remove the last saved parameters and replace with the ones
        saved before this round.
    """
    new_params = parameters
    # Save parameters in clients_params/cid_params
    path_file = f"{params_dir}/{cid}_params.npy"
    if os.path.exists(params_dir) is False:
        os.mkdir(params_dir)
    if os.path.exists(path_file):
        # load old parameters
        old_params = np.load(path_file, allow_pickle=True)
        if remove_last:
            old_params = old_params[:-1]
            if rrl:
                new_params = old_params[-1]
        # add new parameters
        new_params = np.vstack((old_params, new_params))

    # save parameters
    np.save(path_file, new_params)

