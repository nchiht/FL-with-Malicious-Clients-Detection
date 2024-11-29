import math
from typing import Dict, List, Tuple

import numpy as np
from flwr.common import FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


# pylint: disable=unused-argument
def no_attack(
    ordered_results: List[Tuple[ClientProxy, FitRes]], states: Dict[str, bool], **kwargs
):
    """No attack."""
    return ordered_results, {}

def gaussian_attack(ordered_results, states, **kwargs):
    """Apply Gaussian attack on parameters.

    Parameters
    ----------
    ordered_results
        List of tuples (client_proxy, fit_result) ordered by client id.
    states
        Dictionary of client ids and their states (True if malicious, False otherwise).
    magnitude
        Magnitude of the attack.
    dataset_name
        Name of the dataset.

    Returns
    -------
    results
        List of tuples (client_proxy, fit_result) ordered by client id.
    """
    magnitude = kwargs.get("magnitude", 0.0)
    # dataset_name = kwargs.get("dataset_name", "no name")
    results = ordered_results.copy()

    def perturbate(vect):
        return vect + np.random.normal(loc=0, scale=magnitude, size=vect.size)

    for idx, (proxy, fitres) in enumerate(ordered_results):
        if states[proxy.cid]:
            params = parameters_to_ndarrays(fitres.parameters)
            new_params = []
            for par in params:
                # if par is an array of one element, it is a scalar
                if par.size == 1:
                    new_params.append(perturbate(par))
                else:
                    new_params.append(np.apply_along_axis(perturbate, 0, par))
            fitres.parameters = ndarrays_to_parameters(new_params)
            # results[int(fitres.metrics["node_id"])] = (proxy, fitres)
            results[idx] = (proxy, fitres)
    return results, {}
