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
from logging import DEBUG, INFO
from typing import Optional, Callable, Union, List
from sklearn.cluster import KMeans
import torch
import numpy as np
import timeit



# def calculate_gradients(fitres, global_model):
#     gradients = []
#     for i in range(len(fitres.tensors)):
#         fitres_tensor = torch.from_numpy(np.frombuffer(fitres.tensors[i], dtype=np.float32))
#         global_model_tensor = torch.from_numpy(np.frombuffer(global_model.tensors[i], dtype=np.float32))

#         # Pad the smaller tensor with zeros
#         max_size = max(fitres_tensor.numel(), global_model_tensor.numel())
#         fitres_tensor_padded = torch.cat((fitres_tensor, torch.zeros(max_size - fitres_tensor.numel())))
#         global_model_tensor_padded = torch.cat((global_model_tensor, torch.zeros(max_size - global_model_tensor.numel())))
        
#         gradient = fitres_tensor_padded - global_model_tensor_padded
#         gradients.append(gradient.numpy().tobytes())
#     return Parameters(tensors=gradients, tensor_type=fitres.tensor_type)

def detection(score, nobyz):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    
    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred
    real_label=np.ones(100)
    real_label[:nobyz]=0
    acc=len(label_pred[label_pred==real_label])/100
    recall=1-np.sum(label_pred[:nobyz])/nobyz
    fpr=1-np.sum(label_pred[nobyz:])/(100-nobyz)
    fnr=np.sum(label_pred[:nobyz])/nobyz
    # print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    # print(silhouette_score(score.reshape(-1, 1), label_pred))
    # print('defence.py line233 label_pred (0 = malicious pred)', label_pred)
    return label_pred

def detection1(score):
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min)/(max-min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    # print('defense line278 gapDiff:', gapDiff)
    select_k = 2  # default detect attacks
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i+1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1

def fld_distance(old_update_list, local_update_list, net_glob, attack_number, hvp):
    pred_update = []
    distance = []
    for i in range(len(old_update_list)):
        pred_update.append((old_update_list[i] + hvp).view(-1))
        
    
    pred_update = torch.stack(pred_update)
    local_update_list = torch.stack(local_update_list)
    old_update_list = torch.stack(old_update_list)
    
    distance = torch.norm((old_update_list - local_update_list), dim=1)
    # print('defense line219 distance(old_update_list - local_update_list):',distance)
    # auc1 = roc_auc_score(pred_update.numpy(), distance)
    # distance = torch.norm((pred_update - local_update_list), dim=1).numpy()
    # auc2 = roc_auc_score(pred_update.numpy(), distance)
    # print("Detection AUC: %0.4f; Detection AUC: %0.4f" % (auc1, auc2))
    
    # print('defence line 211 pred_update.shape:', pred_update.shape)
    distance = torch.norm((pred_update - local_update_list), dim=1)
    # print('defence line 211 distance.shape:', distance.shape)
    # distance = nn.functional.norm((pred_update - local_update_list), dim=0).numpy()
    distance = distance / torch.sum(distance)
    return distance

def calculate_gradients(fitres_ndarrays, global_model_ndarrays, tensors_type="numpy.ndarray"):
    """
    Calculate the gradients between local fit results and a global model.

    This function computes the difference between each tensor in the local fit 
    results and the corresponding tensor in the global model. The tensors are 
    padded with zeros to match sizes before computing the difference.

    Parameters
    ----------
    fitres_ndarrays : list of np.ndarray
        A list of numpy arrays representing the local fit results.
    global_model_ndarrays : list of np.ndarray
        A list of numpy arrays representing the global model's parameters.
    tensors_type : str, optional
        The type of the tensors to be returned (default is "numpy.ndarray").

    Returns
    -------
    Parameters
        A Parameters object containing the calculated gradients as tensors.
    """
    gradients = []
    for fitres_tensor, global_model_tensor in zip(fitres_ndarrays, global_model_ndarrays):
        # Pad the smaller tensor with zeros
        max_size = max(fitres_tensor.size, global_model_tensor.size)
        fitres_tensor_padded = np.pad(fitres_tensor, (0, max_size - fitres_tensor.size))
        global_model_tensor_padded = np.pad(global_model_tensor, (0, max_size - global_model_tensor.size))
        
        gradient = fitres_tensor_padded - global_model_tensor_padded
        # gradients.append(gradient.tobytes())
        gradients.append(gradient)
    return Parameters(tensors=gradients, tensor_type=tensors_type)

def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def lbfgs_torch(S_k_list, Y_k_list, v):
    curr_S_k = torch.stack(S_k_list)
    curr_S_k = curr_S_k.transpose(0, 1).cpu() #(10,xxxxxx)
    # print('------------------------')
    # print('curr_S_k.shape', curr_S_k.shape)
    curr_Y_k = torch.stack(Y_k_list)
    curr_Y_k = curr_Y_k.transpose(0, 1).cpu() #(10,xxxxxx)
    S_k_time_Y_k = curr_S_k.transpose(0, 1) @ curr_Y_k
    S_k_time_Y_k = S_k_time_Y_k.cpu()


    S_k_time_S_k = curr_S_k.transpose(0, 1) @ curr_S_k
    S_k_time_S_k = S_k_time_S_k.cpu()
    # print('S_k_time_S_k.shape', S_k_time_S_k.shape)
    R_k = np.triu(S_k_time_Y_k.numpy())
    L_k = S_k_time_Y_k - torch.from_numpy(R_k).cpu()
    sigma_k = Y_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1) / (S_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1))
    sigma_k=sigma_k.cpu()
    
    D_k_diag = S_k_time_Y_k.diagonal()
    upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = torch.cat([L_k.transpose(0, 1), -D_k_diag.diag()], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat_inv = mat.inverse()
    # print('mat_inv.shape',mat_inv.shape)
    v = v.view(-1,1).cpu()

    approx_prod = sigma_k * v
    # print('approx_prod.shape',approx_prod.shape)
    # print('v.shape',v.shape)
    # print('sigma_k.shape',sigma_k.shape)
    # print('sigma_k',sigma_k)
    p_mat = torch.cat([curr_S_k.transpose(0, 1) @ (sigma_k * v), curr_Y_k.transpose(0, 1) @ v], dim=0)
    
    approx_prod -= torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1) @ mat_inv @ p_mat
    # print('approx_prod.shape',approx_prod.shape)
    # print('approx_prod.shape',approx_prod.shape)
    # print('approx_prod.shape.T',approx_prod.T.shape)

    return approx_prod.T

class EnhancedServer(Server):
    def __init__(
            self, 
            *, 
            client_manager: ClientManager = SimpleClientManager(), 
            strategy: Strategy = FedAvg(), #
            sampling: int = 0, 
            history_dir: str = "data/clients_params",
            warmup_rounds: int = 2,
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

        # initialize storing variables
        self.old_update_list = []
        self.weight_record = []
        self.update_record = []
        self.malicious_score = torch.zeros(self._client_manager.num_available())
        self.last_weight = torch.tensor([])

        # # initialize storing variables
        # self.old_update_list = []
        # self.weight_record = []
        # self.update_record = []
        # self.malicious_score = torch.zeros(self._client_manager.num_available())
        # self.last_weight = torch.tensor([])

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
        log(DEBUG, "Weight updates: %s", gradient_updates)

        current_global_weight = torch.tensor(flatten_params(parameters_to_ndarrays(self.parameters)))
        local_update_list = [local for _, local in gradient_updates.items()]

        # Detect malicious clients using FLDetector
        if server_round > self.warmup_rounds + 1:
            # self.sampling = 0
            log(DEBUG, "Starting to detect malicious clients")

            # Calculate the distance between the global model and the local model
            hvp = lbfgs_torch(self.weight_record, self.update_record, current_global_weight - self.last_weight) 
            log(DEBUG, "hvp: %s", hvp)
            # distance = fld_distance(old_update_list, gradient_updates, None, None, hvp)
            distance = fld_distance(self.old_update_list, local_update_list, None, None, hvp)
            distance = distance.view(1,-1)
            log(DEBUG, "distance: %s", distance)
            self.malicious_score = torch.cat((self.malicious_score, distance), dim=0)
            log(DEBUG, "self.malicious_score: %s", self.malicious_score)
            if self.malicious_score.shape[0] > self.warmup_rounds+1:
                # if detection1(np.sum(self.malicious_score[-self.warmup_rounds:].numpy(), axis=0)):
                #     label = detection(np.sum(self.malicious_score[-self.warmup_rounds:].numpy(), axis=0), 1)
                # else:
                #     label = np.ones(100)
                # selected_client = []
                # for client in range(100):
                #     if label[client] == 1:
                #         selected_client.append(client)
                # new_w_glob = FedAvg([w_locals[client] for client in selected_client])
                log(DEBUG, "Detecting malicious clients")

                # Aggregate training results
                aggregated_result: tuple[
                    Optional[Parameters],
                    dict[str, Scalar],
                ] = self.strategy.aggregate_fit(server_round, results, failures)
            else:
                # Aggregate training results
                aggregated_result: tuple[
                    Optional[Parameters],
                    dict[str, Scalar],
                ] = self.strategy.aggregate_fit(server_round, results, failures)
        else:
            hvp = None
            # Aggregate training results
            aggregated_result: tuple[
                Optional[Parameters],
                dict[str, Scalar],
            ] = self.strategy.aggregate_fit(server_round, results, failures)
            

        parameters_aggregated, metrics_aggregated = aggregated_result
        # new_global_weight = torch.tensor(flatten_params(parameters_to_ndarrays(parameters_aggregated)))
        update = calculate_gradients(parameters_to_ndarrays(self.parameters), parameters_to_ndarrays(parameters_aggregated))  #w_t+1 = w_t - a*g_t => g_t = w_t - w_t+1 (a=1)
        update = torch.tensor(flatten_params(update.tensors))

        if server_round > 1:
            # log(DEBUG, "Gap: %s", current_global_weight.cpu() - self.last_weight.cpu())
            self.weight_record.append(current_global_weight.cpu() - self.last_weight.cpu())
            self.update_record.append(update.cpu() -  self.last_update.cpu())
        if server_round > self.warmup_rounds: 
            del self.weight_record[0]
            del self.update_record[0]

        # log(DEBUG, "weight record: %s", self.weight_record)
        # log(DEBUG, "update record: %s", self.update_record)
        # log(DEBUG, "current global weight: %s", current_global_weight)
        # log(DEBUG, "last weight: %s", self.last_weight)

        self.last_update = update
        self.old_update_list = local_update_list
        self.last_weight = current_global_weight

        # log(DEBUG, "last weight after: %s", self.last_weight)

        return parameters_aggregated, metrics_aggregated, (results, failures)

        
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