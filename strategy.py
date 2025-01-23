from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, Krum, Strategy
from flwr.server.strategy.aggregate import aggregate_krum
from logging import WARNING
from typing import Callable, Optional, Union
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg

from flwr.common import (
    FitRes,
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
import torch


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
    
    def configure_fit(
        self, server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager,
        warmup_rounds: int = 0,
        flags = None
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {
            "server_round": server_round,
            "warmup_rounds": warmup_rounds,
            "dp_flags": False,
            "learning_rate": 0.001
            }
        # if server_round >= 30:
        #     config["learning_rate"] = 0.003
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        final_results = []
        results = [(client, fit_ins) for client in clients]
        for idx, (client, fit_ins) in enumerate(results):
            config = dict(fit_ins.config)
            config["dp_flags"] = flags[idx]
            config["index"] = idx
            new_fitins = FitIns(fit_ins.parameters, config)
            # log(WARNING, "client index %s: %s/%s", idx, new_fitins.config["dp_flags"], flags[idx])
            final_results.append((client, new_fitins))
        # Return client/config pairs
        return final_results
        # return [(client, fit_ins) for client in clients]


class CustomStrategy_FedAvg(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        # num_malicious_clients: int = 0,
        # num_clients_to_keep: int = 0, # Multi Krum
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
        inplace: bool = True,
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
            inplace = inplace
        )
        # self.num_malicious_clients = num_malicious_clients
        # self.num_clients_to_keep = num_clients_to_keep

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAvg(accept_failures={self.accept_failures})"
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

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
    def configure_fit(
        self, server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager,
        warmup_rounds: int = 0,
        flags = None
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {
            "server_round": server_round,
            "warmup_rounds": warmup_rounds,
            "dp_flags": False,
            "learning_rate": 0.001
            }
        # if server_round >= 30:
        #     config["learning_rate"] = 0.003
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        final_results = []
        results = [(client, fit_ins) for client in clients]
        for idx, (client, fit_ins) in enumerate(results):
            config = dict(fit_ins.config)
            config["dp_flags"] = flags[idx]
            config["index"] = idx
            new_fitins = FitIns(fit_ins.parameters, config)
            # log(WARNING, "client index %s: %s/%s", idx, new_fitins.config["dp_flags"], flags[idx])
            final_results.append((client, new_fitins))
        # Return client/config pairs
        return final_results
        # return [(client, fit_ins) for client in clients]

# class EnhancedStrategy_Krum(FedAvg):
#     def __init__(
#         self,
#         *,
#         fraction_fit: float = 1.0,
#         fraction_evaluate: float = 1.0,
#         min_fit_clients: int = 2,
#         min_evaluate_clients: int = 2,
#         min_available_clients: int = 2,
#         num_malicious_clients: int = 0,
#         num_clients_to_keep: int = 0, # Multi Krum
#         evaluate_fn: Optional[
#             Callable[
#                 [int, NDArrays, dict[str, Scalar]],
#                 Optional[tuple[float, dict[str, Scalar]]],
#             ]
#         ] = None,
#         on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
#         on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
#         accept_failures: bool = True,
#         initial_parameters: Optional[Parameters] = None,
#         fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
#         evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
#     ) -> None:
#         super().__init__(
#             fraction_fit=fraction_fit,
#             fraction_evaluate=fraction_evaluate,
#             min_fit_clients=min_fit_clients,
#             min_evaluate_clients=min_evaluate_clients,
#             min_available_clients=min_available_clients,
#             evaluate_fn=evaluate_fn,
#             on_fit_config_fn=on_fit_config_fn,
#             on_evaluate_config_fn=on_evaluate_config_fn,
#             accept_failures=accept_failures,
#             initial_parameters=initial_parameters,
#             fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
#             evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
#         )
#         self.num_malicious_clients = num_malicious_clients
#         self.num_clients_to_keep = num_clients_to_keep

#     def __repr__(self) -> str:
#         """Compute a string representation of the strategy."""
#         rep = f"Krum(accept_failures={self.accept_failures})"
#         return rep

#     def aggregate_fit(
#         self,
#         server_round: int,
#         results: list[tuple[ClientProxy, FitRes]],
#         failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
#     ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
#         """Aggregate fit results using Krum."""
#         if not results:
#             return None, {}
#         # Do not aggregate if there are failures and failures are not accepted
#         if not self.accept_failures and failures:
#             return None, {}

#         # Convert results
#         weights_results = [
#             (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
#             for _, fit_res in results
#         ]

#         # TODO: add FLDetector

#         parameters_aggregated = ndarrays_to_parameters(
#             aggregate_krum(
#                 weights_results, self.num_malicious_clients, self.num_clients_to_keep
#             )
#         )

#         # Aggregate custom metrics if aggregation fn was provided
#         metrics_aggregated = {}
#         if self.fit_metrics_aggregation_fn:
#             fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
#             metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
#         elif server_round == 1:  # Only log this warning once
#             log(WARNING, "No fit_metrics_aggregation_fn provided")

#         return parameters_aggregated, metrics_aggregated
    
#     def configure_fit(
#         self, server_round: int, 
#         parameters: Parameters, 
#         client_manager: ClientManager,
#         warmup_rounds: int = 0,
#         flags = None
#     ) -> list[tuple[ClientProxy, FitIns]]:
#         """Configure the next round of training."""
#         config = {
#             "server_round": server_round,
#             "warmup_rounds": warmup_rounds,
#             "dp_flags": False,
#             "learning_rate": 0.001
#             }
#         if server_round >= 30:
#             config["learning_rate"] = 0.003
#         if self.on_fit_config_fn is not None:
#             # Custom fit config function provided
#             config = self.on_fit_config_fn(server_round)
#         fit_ins = FitIns(parameters, config)

#         # Sample clients
#         sample_size, min_num_clients = self.num_fit_clients(
#             client_manager.num_available()
#         )
#         clients = client_manager.sample(
#             num_clients=sample_size, min_num_clients=min_num_clients
#         )
#         final_results = []
#         results = [(client, fit_ins) for client in clients]
#         for idx, (client, fit_ins) in enumerate(results):
#             config = dict(fit_ins.config)
#             config["dp_flags"] = flags[idx]
#             config["index"] = idx
#             new_fitins = FitIns(fit_ins.parameters, config)
#             # log(WARNING, "client index %s: %s/%s", idx, new_fitins.config["dp_flags"], flags[idx])
#             final_results.append((client, new_fitins))
#         # Return client/config pairs
#         return final_results
#         # return [(client, fit_ins) for client in clients]