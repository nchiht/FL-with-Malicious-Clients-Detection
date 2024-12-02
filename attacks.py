import math
from typing import Dict, List, Tuple

import numpy as np
from flwr.common import FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from scipy.stats import norm


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
        
        """Add Gaussian noise based on mean and std of vect."""
        mean = np.mean(vect)
        std_dev = np.std(vect)
        noise = np.random.normal(loc=mean, scale=magnitude * std_dev, size=vect.shape)
        return vect + noise
        # return vect + np.random.normal(loc=0, scale=magnitude, size=vect.size)

    for idx, (proxy, fitres) in enumerate(ordered_results):
        if states[fitres.metrics["partition_id"]]:
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

##chưa config-------------------------------------------------
# def lie_attack(
#     ordered_results,
#     states,
#     omniscent=True,
#     **kwargs,
# ):
#     """Apply Omniscent LIE attack, Baruch et al. (2019) on parameters.

#     Parameters
#     ----------
#     ordered_results
#         List of tuples (client_proxy, fit_result) ordered by client id.
#     states
#         Dictionary of client ids and their states (True if malicious, False otherwise).
#     omniscent
#         Whether the attacker knows the local models of all clients or not.

#     Returns
#     -------
#     results
#         List of tuples (client_proxy, fit_result) ordered by client id.
#     """
#     results = ordered_results.copy()
#     params = [parameters_to_ndarrays(fitres.parameters) for _, fitres in results]
#     grads_mean = [np.mean(layer, axis=0) for layer in zip(*params)]
#     grads_stdev = [np.std(layer, axis=0) ** 0.5 for layer in zip(*params)]

#     if not omniscent:
#         # if not omniscent, the attacker doesn't know the
#         # local models of all clients, but only of the corrupted ones
#         params = [
#             params[i]
#             for i in range(len(params))
#             if states[results[i][1].metrics["cid"]]
#         ]

#     num_clients = len(ordered_results)
#     num_malicious = sum(val is True for val in states.values())

#     # pylint: disable=c-extension-no-member
#     num_supporters = math.floor((num_clients / 2) + 1) - num_malicious

#     z_max = norm.cdf(
#         (num_clients - num_malicious - num_supporters) / (num_clients - num_malicious)
#     )

#     for proxy, fitres in ordered_results:
#         if states[fitres.metrics["cid"]]:
#             mul_std = [layer * z_max for layer in grads_stdev]
#             new_params = [grads_mean[i] - mul_std[i] for i in range(len(grads_mean))]
#             fitres.parameters = ndarrays_to_parameters(new_params)
#             results[int(fitres.metrics["cid"])] = (proxy, fitres)
#     return results, {}
