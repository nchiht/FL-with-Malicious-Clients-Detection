from flwr.server.strategy import FedAvg, Krum, Strategy
from flwr.server.strategy.aggregate import aggregate_krum
from logging import WARNING
from typing import Callable, Optional, Union

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
import torch


import torch

# def l_bfgs(args, S_k_list, Y_k_list, v):
#     device = torch.device("cpu")
    
#     # Concatenate S_k and Y_k along columns
#     curr_S_k = torch.cat(S_k_list, dim=1).to(device)
#     curr_Y_k = torch.cat(Y_k_list, dim=1).to(device)
    
#     # Compute matrix products
#     S_k_time_Y_k = torch.matmul(curr_S_k.T, curr_Y_k)
#     S_k_time_S_k = torch.matmul(curr_S_k.T, curr_S_k)
    
#     # Extract upper triangular matrix (R_k)
#     R_k = torch.triu(S_k_time_Y_k)
#     L_k = S_k_time_Y_k - R_k
    
#     # Compute sigma_k
#     sigma_k = torch.matmul(Y_k_list[-1].T, S_k_list[-1]) / torch.matmul(S_k_list[-1].T, S_k_list[-1])
    
#     # Compute diagonal matrix (D_k_diag)
#     D_k_diag = torch.diag(S_k_time_Y_k)
    
#     # Construct upper and lower parts of the matrix
#     upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
#     lower_mat = torch.cat([L_k.T, -torch.diag(D_k_diag)], dim=1)
#     mat = torch.cat([upper_mat, lower_mat], dim=0)
    
#     # Invert the matrix
#     mat_inv = torch.linalg.inv(mat)
    
#     # Compute the approximate product
#     approx_prod = sigma_k * v.to(device)
#     p_mat = torch.cat([torch.matmul(curr_S_k.T, sigma_k * v), torch.matmul(curr_Y_k.T, v)], dim=0)
#     approx_prod -= torch.matmul(torch.matmul(torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)
    
#     return approx_prod

class EnhancedStrategy(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        num_malicious_clients: int = 0,
        num_clients_to_keep: int = 0, # Multi Krum
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"Krum(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using Krum."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # TODO: add FLDetector

        parameters_aggregated = ndarrays_to_parameters(
            aggregate_krum(
                weights_results, self.num_malicious_clients, self.num_clients_to_keep
            )
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated