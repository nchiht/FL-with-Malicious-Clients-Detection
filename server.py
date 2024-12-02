from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.strategy import FedAvg, Strategy
from flwr.server.history import History
from flwr.server.server import fit_clients, Server

from flwr.common import (
    Metrics, 
    Context, 
    Parameters, 
    Scalar, 
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.common.logger import log
from utils.util import flatten_params, save_params
from utils.evaluation import update_confusion_matrix
from logging import DEBUG, INFO
from typing import Optional, Callable, Union, List
import numpy as np
import timeit


class EnhancedServer(Server):
    def __init__(
            self, 
            *, 
            client_manager: ClientManager = SimpleClientManager(), 
            strategy: Strategy = FedAvg(), #
            sampling: int = 0, 
            history_dir: str = "clients_params",
            warmup_rounds: int = 1,
            num_malicious: int = 3,
            attack_fn: Callable,
            magnitude: float = 0.0
        ) -> None:

        super().__init__(
            client_manager=client_manager, 
            strategy=strategy
        )
        self.strategy = strategy
        self.sampling = sampling
        self.history_dir = history_dir
        self.malicious_lst: List = []
        self.aggregated_parameters: List = []
        self.warmup_rounds = warmup_rounds
        self.num_malicious = num_malicious
        self.attack_fn = attack_fn
        self.magnitude = magnitude
        self.clients_state = {}

    def fit(self, num_rounds, timeout):
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)

        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            # res[1]["TP"] = 0
            # res[1]["TN"] = 0
            # res[1]["FP"] = 0
            # res[1]["FN"] = 0
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                # Update confusion matrix

                # self.confusion_matrix = update_confusion_matrix(
                #     self.confusion_matrix,
                #     self.clients_state,
                #     self.malicious_clients_idx,
                #     self.good_clients_idx,
                # )
                # metrics_cen["TP"] = self.confusion_matrix["TP"]
                # metrics_cen["TN"] = self.confusion_matrix["TN"]
                # metrics_cen["FP"] = self.confusion_matrix["FP"]
                # metrics_cen["FN"] = self.confusion_matrix["FN"]
                log(
                    INFO,
                    '''Fit progress: 
                    \t- Current round: %s, 
                    \t- Loss: %s, 
                    \t- Metrics: %s, 
                    \t- Time: %s''',
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history, elapsed

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ):
        # pylint: disable-msg=R0912
        """Perform a single round of federated learning."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            INFO,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )
        
        # Randomly decide which client is malicious
        # Chỉ chọn malicious clients ở round đầu tiên
        if server_round == (self.warmup_rounds + 1):
            size = self.num_malicious
            log(INFO, "Selecting %s malicious clients", size)
            self.malicious_lst = np.random.choice(
                range(self._client_manager.num_available()), size=size, replace=False
            )

        # Create dict clients_state to keep track of malicious clients
        # clients_state = dict()
        for idx, (proxy, _) in enumerate(client_instructions):
            self.clients_state[idx] = False
            if idx in self.malicious_lst:
                self.clients_state[idx] = True
        # Sort clients states
        self.clients_state = {k: self.clients_state[k] for k in sorted(self.clients_state)}
        log(
            DEBUG,
            "fit_round %s: malicious clients selected %s, clients_state %s",
            server_round,
            self.malicious_lst,
            self.clients_state
        )

        clients_state = self.clients_state
        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round
        )
        log(
            INFO,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
        
        #check
        # list_id = [(proxy.cid, fitres.metrics["partition_id"]) for proxy, fitres in results]
        # log(INFO, "List of clients' id and partition id: %s", list_id)

        # Save parameters of each client as time series
        ordered_results = [0 for _ in range(len(results))]
        for idx, (proxy, fitres) in enumerate(results):
            params = flatten_params(parameters_to_ndarrays(fitres.parameters))
            if self.sampling > 0:
                # if the sampling number is greater than the number of
                # parameters, just sample all of them
                self.sampling = min(self.sampling, len(params))
                if len(self.params_indexes) == 0:
                    # Sample a random subset of parameters
                    self.params_indexes = np.random.randint(
                        0, len(params), size=self.sampling
                    )

                params = params[self.params_indexes]

            save_params(params, fitres.metrics["partition_id"], params_dir=self.history_dir)

            # Re-arrange results in the same order as clients' cids impose
            # ordered_results[int(fitres.metrics["node_id"])] = (proxy, fitres)
            ordered_results[idx] = (proxy, fitres)

        log(INFO, "Clients state: %s", clients_state)

        # Initialize aggregated_parameters if it is the first round
        if self.aggregated_parameters == []:
            for key, val in clients_state.items():
                if val is False:
                    # for idx, (proxy, fitres) in enumerate(ordered_results):
                    #     if proxy.cid == key:
                            # log(INFO, "Aggregated parameters initialized: %s, %s", proxy.cid, fitres.metrics["node_id"])
                            self.aggregated_parameters = parameters_to_ndarrays(
                                # ordered_results[idx][1].parameters
                                ordered_results[int(key)][1].parameters
                            )
                            break
                    # self.aggregated_parameters = parameters_to_ndarrays(
                    #     ordered_results[int(key)][1].parameters
                    # )
                    # break

        # Apply attack function
        # the server simulates an attacker that controls a fraction of the clients
        # Client state có key là partition_id, value là True nếu là malicious, False nếu là benign
        if self.attack_fn is not None and server_round > self.warmup_rounds:
            log(INFO, "Applying attack function")
            results, others = self.attack_fn(
                ordered_results,
                clients_state,
                # omniscent=self.omniscent,
                magnitude=self.magnitude,
                # w_re=self.aggregated_parameters,
                # threshold=self.threshold,
                d=len(self.aggregated_parameters),
                # dataset_name=self.dataset_name,
                # to_keep=self.to_keep,
                malicious_num=self.num_malicious,
                # num_layers=len(self.aggregated_parameters),
            )

            # Update saved parameters time series after the attack
            for proxy, fitres in results:
                if clients_state[fitres.metrics["partition_id"]]:
                    if self.sampling > 0:
                        params = flatten_params(
                            parameters_to_ndarrays(fitres.parameters)
                        )[self.params_indexes]
                    else:
                        params = flatten_params(
                            parameters_to_ndarrays(fitres.parameters)
                        )
                    log(
                        INFO,
                        "Saving parameters of client %s with shape %s after the attack",
                        fitres.metrics["partition_id"],
                        params.shape,
                    )
                    save_params(
                        params,
                        fitres.metrics["partition_id"],
                        params_dir=self.history_dir,
                        remove_last=True,
                    )
        else:
            results = ordered_results
            others = {}
    
        # good_clients_idx = []
        # malicious_clients_idx = []
        # if isinstance(self.strategy, Flanders):
        #     # Aggregate training results
        #     log(INFO, "fit_round - Aggregating training results")
        #     aggregated_result = self.strategy.aggregate_fit(server_round, results, failures, clients_state)
        #     (
        #         parameters_aggregated,
        #         metrics_aggregated,
        #         good_clients_idx,
        #         malicious_clients_idx,
        #     ) = aggregated_result
        #     log(INFO, "Malicious clients: %s", malicious_clients_idx)

        #     log(INFO, "clients_state: %s", clients_state)

        #     # For clients detected as malicious, replace the last params in
        #     # their history with tha current global model, otherwise the
        #     # forecasting in next round won't be reliable (see the paper for
        #     # more details)
        #     if server_round > self.warmup_rounds:
        #         log(INFO, "Saving parameters of clients")
        #         for idx in malicious_clients_idx:
        #             if self.sampling > 0:
        #                 new_params = flatten_params(
        #                     parameters_to_ndarrays(parameters_aggregated)
        #                 )[self.params_indexes]
        #             else:
        #                 new_params = flatten_params(
        #                     parameters_to_ndarrays(parameters_aggregated)
        #                 )

        #             print(f"Saving parameters of client {idx} with shape {new_params.shape}")
        #             save_params(
        #                 new_params,
        #                 idx,
        #                 params_dir=self.history_dir,
        #                 remove_last=True,
        #                 rrl=False,
        #             )
        # elif isinstance(self.strategy, DnC):
        #     log(INFO, "fit_round - Aggregating training results")
        #     aggregated_result = self.strategy.aggregate_fit(server_round, results, failures, clients_state)
        #     parameters_aggregated, metrics_aggregated = aggregated_result
        # else:
        #     # Aggregate training results
        #     log(INFO, "fit_round - Aggregating training results")
        #     aggregated_result = self.strategy.aggregate_fit(server_round, results, failures)
        #     parameters_aggregated, metrics_aggregated = aggregated_result

        # self.clients_state = clients_state
        # self.good_clients_idx = good_clients_idx
        # self.malicious_clients_idx = malicious_clients_idx
        # return parameters_aggregated, metrics_aggregated, (results, failures)
        
        # # Collect `fit` results from all clients participating in this round
        # results, failures = fit_clients(
        #     client_instructions=client_instructions,
        #     max_workers=self.max_workers,
        #     timeout=timeout,
        #     group_id=server_round,
        # )
        # log(
        #     INFO,
        #     "aggregate_fit: received %s results and %s failures",
        #     len(results),
        #     len(failures),
        # )

        # Aggregate training results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)