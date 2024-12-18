import os   
os.environ['RAY_DEDUP_LOGS'] = '0'
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
    ndarrays_to_parameters,
    bytes_to_ndarray
)
from flwr.common.logger import log
from utils.util import flatten_params, save_params
from utils.evaluation import update_confusion_matrix
from utils.fldetector import (
    calculate_gradients, 
    fld_distance, 
    detection, 
    detection1,
    lbfgs_torch
)

from logging import DEBUG, INFO
from typing import Optional, Callable, Union, List
import torch
import numpy as np
import timeit
import pandas as pd
import datetime 

class Hist(History):
    def __init__(self):
        super().__init__()
        self.confusion_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    def to_dataframes(self, file_path: str):
        """Convert losses and metrics to pandas DataFrames."""
        # Convert losses_centralized to DataFrame
        df_losses_centralized = pd.DataFrame(self.losses_centralized, columns=["round", "loss"])
        df_losses_centralized.to_csv(file_path + "/losses_centralized.csv", index=False)
        # Convert metrics_centralized to DataFrame
        metrics_data = []
        for key, values in self.metrics_centralized.items():
            for round, value in values:
                metrics_data.append({"round": round, "accuracy": value})
        df_metrics_centralized = pd.DataFrame(metrics_data)
        df_metrics_centralized.to_csv(file_path + "/metrics_centralized.csv", index=False)
        log(INFO, "DataFrames saved")
        return df_losses_centralized, df_metrics_centralized
    
class EnhancedServer(Server):
    def __init__(
            self, 
            *, 
            client_manager: ClientManager = SimpleClientManager(), 
            strategy: Strategy = FedAvg(), #
            sampling: int = 0, 
            history_dir: str = "data/clients_params",
            warmup_rounds: int = 1,
            num_malicious: int = 3,
            window_size: int = 3,
            attack_fn: Callable,
            magnitude: float = 0.0,
            num_data_poisoning: int = 2,
            defense: bool = True,
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
        self.window_size = window_size
        self.num_data_poisoning = num_data_poisoning
        self.list_data_poisoning = []
        self.flags_data_poisoning = {}
        self.defense = defense

        # initialize storing variables
        self.old_update_list = []
        self.weight_record = []
        self.update_record = []
        self.malicious_score = torch.zeros(self._client_manager.num_available())
        self.last_weight = torch.tensor([])

        self.metrics_detection_df = []

    def fit(self, num_rounds, timeout):
        """Run federated averaging for a number of rounds."""
        history = Hist()

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

        # size = self.num_malicious
        log(INFO, "Selecting %s malicious clients", self.num_malicious + self.num_data_poisoning)
        self.malicious_lst = np.random.choice(
            range(self._client_manager.num_available()), size=self.num_malicious, replace=False
        )
        # Create a set of available clients excluding the malicious clients
        available_clients = set(range(self._client_manager.num_available())) - set(self.malicious_lst)
        self.list_data_poisoning = np.random.choice(
            list(available_clients), size=self.num_data_poisoning, replace=False
        )

        # Create dict clients_state and flags for data poison to keep track of malicious clients
        # clients_state = dict()
        for idx, _ in enumerate(range(self._client_manager.num_available())):
            self.clients_state[idx] = False
            self.flags_data_poisoning[idx] = False

            if idx in self.malicious_lst:
                self.clients_state[idx] = True
            if idx in self.list_data_poisoning:
                self.flags_data_poisoning[idx] = True
        log(INFO, "Num available clients: %s", self._client_manager.num_available())
        # Sort clients states
        self.clients_state = {k: self.clients_state[k] for k in sorted(self.clients_state)}
        self.flags_data_poisoning = {k: self.flags_data_poisoning[k] for k in sorted(self.flags_data_poisoning)}
        
        log(
            DEBUG,
            """
            malicious clients selected %s\n
            clients_state: %s
            data poisoning: %s
            flags of data poisoning: %s
            """,
            self.malicious_lst,
            self.clients_state,
            self.list_data_poisoning,
            self.flags_data_poisoning
        )

        # clients_state = self.clients_state
        # flags_data_poisoning = self.flags_data_poisoning

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

        detection_df = pd.DataFrame(
            self.metrics_detection_df,
            columns=["server_round", "accuracy", "recall", "f1_score", "precision", "confusion_matrix"]
        )
        base_path = f"data/metrics/run_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if (os.path.exists(base_path) == False):
            os.makedirs(base_path)

        detection_df.to_csv(base_path + "/detection_metrics.csv", index=False)
        log(INFO, "Detection metrics saved")

        history.to_dataframes(file_path=base_path)

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
            warmup_rounds=self.warmup_rounds,
            flags=self.flags_data_poisoning
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

        log(DEBUG, "Model poisoning: %s", self.malicious_lst)
        log(DEBUG, "Data poisoning: %s", self.list_data_poisoning)

        # Save parameters of each client as time series
        ordered_results = [0 for _ in range(len(results))]
        for idx, (proxy, fitres) in enumerate(results):
            # params = flatten_params(parameters_to_ndarrays(fitres.parameters))
            # if self.sampling > 0:
            #     # if the sampling number is greater than the number of
            #     # parameters, just sample all of them
            #     self.sampling = min(self.sampling, len(params))
            #     if len(self.params_indexes) == 0:
            #         # Sample a random subset of parameters
            #         self.params_indexes = np.random.randint(
            #             0, len(params), size=self.sampling
            #         )

            #     params = params[self.params_indexes]

            # save_params(params, fitres.metrics["partition_id"], params_dir=self.history_dir)

            # Re-arrange results in the same order as clients' cids impose
            # ordered_results[int(fitres.metrics["node_id"])] = (proxy, fitres)
            ordered_results[idx] = (proxy, fitres)

        # Initialize aggregated_parameters if it is the first round
        if self.aggregated_parameters == []:
            for key, val in self.clients_state.items():
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
                self.clients_state,
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
                if self.clients_state[fitres.metrics["partition_id"]]:
                    # if self.sampling > 0:
                    #     params = flatten_params(
                    #         parameters_to_ndarrays(fitres.parameters)
                    #     )[self.params_indexes]
                    # else:
                    #     params = flatten_params(
                    #         parameters_to_ndarrays(fitres.parameters)
                    #     )
                    log(
                        INFO,
                        "Saving parameters of client %s after the attack",
                        fitres.metrics["partition_id"]
                    )
                    # save_params(
                    #     params,
                    #     fitres.metrics["partition_id"],
                    #     params_dir=self.history_dir,
                    #     remove_last=True,
                    # )
        else:
            results = ordered_results
            others = {}
    
        # Get gradients
        gradient_updates = {}
        for idx, (proxy, fitres) in enumerate(results):
            gradient = calculate_gradients(
                parameters_to_ndarrays(fitres.parameters), 
                parameters_to_ndarrays(self.parameters), 
                tensors_type=fitres.parameters.tensor_type
            ) # Calculate the gradient of the client's parameters with respect to the global model parameters (fitres.parameters, self.parameters)
            gradient_updates[fitres.metrics["partition_id"]] = -1*torch.tensor(flatten_params(gradient.tensors)).cpu() 
            # Add the weight update to the list, multiplied by -1 to make it a gradient
        gradient_updates = {k: gradient_updates[k] for k in sorted(gradient_updates)}
        # log(DEBUG, "Weight updates: %s", gradient_updates)

        current_global_weight = torch.tensor(flatten_params(parameters_to_ndarrays(self.parameters)))
        local_update_list = [local for _, local in gradient_updates.items()]
         
        # Detect malicious clients using FLDetector
        if (server_round > self.window_size + 1) and (self.defense):
            # self.sampling = 0
            log(DEBUG, "Starting to detect malicious clients")

            # Calculate the distance between the global model and the local model
            hvp = lbfgs_torch(self.weight_record, self.update_record, current_global_weight - self.last_weight) 
            distance = fld_distance(self.old_update_list, local_update_list, None, None, hvp)
            distance = distance.view(1,-1)
            self.malicious_score = torch.cat((self.malicious_score, distance), dim=0)

            if self.malicious_score.shape[0] >= self.window_size:
                log(DEBUG, "Detecting malicious clients")
                check_attack, select_k = detection1(np.sum(self.malicious_score[-self.window_size:].numpy(), axis=0))
                if check_attack:
                    label, metrics_detection = detection(
                                                    np.sum(self.malicious_score[-self.window_size:].numpy(), axis=0), 
                                                    self.clients_state,
                                                    self.flags_data_poisoning,
                                                    steps=select_k - 1
                                                )

                    metrics_detection["server_round"] = server_round
                    data = [
                        metrics_detection["server_round"],
                        metrics_detection["accuracy"],
                        metrics_detection["recall"],
                        metrics_detection["f1"],
                        metrics_detection["precision"],
                        metrics_detection["confusion_matrix"]
                    ]
                    log(DEBUG, "Data: %s", data)
                    self.metrics_detection_df.append(data)
                    log(DEBUG, "Detection metrics: %s", self.metrics_detection_df)

                else:
                    label = np.ones(self._client_manager.num_available())

                selected_clients = []
                for client in range(self._client_manager.num_available()):
                    if label[client] == 1:
                        for index in range(len(results)):
                            if results[index][1].metrics["partition_id"] == client:
                                selected_clients.append(results[index])
                                continue
                        
                log(DEBUG, "Nunber of aggregated clients: %s", len(selected_clients))
                # Aggregate training results
                aggregated_result: tuple[
                    Optional[Parameters],
                    dict[str, Scalar],
                ] = self.strategy.aggregate_fit(server_round, selected_clients, failures)
            else:
                log(DEBUG, "Nunber of aggregated clients: %s", len(results))
                # Aggregate training results
                aggregated_result: tuple[
                    Optional[Parameters],
                    dict[str, Scalar],
                ] = self.strategy.aggregate_fit(server_round, results, failures)
        else:
            log(DEBUG, "Nunber of aggregated clients: %s", len(results))
            hvp = None
            # Aggregate training results
            aggregated_result: tuple[
                Optional[Parameters],
                dict[str, Scalar],
            ] = self.strategy.aggregate_fit(server_round, results, failures)
            
        parameters_aggregated, metrics_aggregated = aggregated_result
        #gradient global
        update = calculate_gradients(parameters_to_ndarrays(self.parameters), parameters_to_ndarrays(parameters_aggregated))  #w_t+1 = w_t - a*g_t => g_t = w_t - w_t+1 (a=1)
        update = torch.tensor(flatten_params(update.tensors))

        if server_round > 1:
            #deltaW
            self.weight_record.append(current_global_weight.cpu() - self.last_weight.cpu())
            #deltaG
            self.update_record.append(update.cpu() -  self.last_update.cpu())
        if server_round > self.window_size + 1: 
            del self.weight_record[0]
            del self.update_record[0]

        self.last_update = update
        self.old_update_list = local_update_list
        self.last_weight = current_global_weight

        return parameters_aggregated, metrics_aggregated, (results, failures)
