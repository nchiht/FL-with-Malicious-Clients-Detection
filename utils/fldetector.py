from flwr.common import Parameters
from flwr.common.logger import log
from logging import DEBUG, INFO, WARNING
from typing import Optional, Callable, Union, List
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_score
import torch
import numpy as np

def convert_clients_state_to_array(clients_state):
    # Initialize an array with the same length as the number of clients
    num_clients = len(clients_state)
    clients_array = np.ones(num_clients)

    # Set the value to 0 for Byzantine clients (True status)
    for index, status in clients_state.items():
        if status:
            clients_array[index] = 0

    return clients_array.astype(int)

def lbfgs_torch(S_k_list, Y_k_list, v):
    curr_S_k = torch.stack(S_k_list)
    curr_S_k = curr_S_k.transpose(0, 1).cpu() #(10,xxxxxx) 
    # print('------------------------')
    # print('curr_S_k.shape', curr_S_k.shape)
    curr_Y_k = torch.stack(Y_k_list)
    curr_Y_k = curr_Y_k.transpose(0, 1).cpu() #(10,xxxxxx) 
    curr_S_k = curr_S_k.double()
    curr_Y_k = curr_Y_k.double()
    S_k_time_Y_k = curr_S_k.transpose(0, 1) @ curr_Y_k 
    S_k_time_Y_k = S_k_time_Y_k.cpu()

    S_k_time_S_k = curr_S_k.transpose(0, 1) @ curr_S_k 
    S_k_time_S_k = S_k_time_S_k.cpu()
    # print('S_k_time_S_k.shape', S_k_time_S_k.shape)
    R_k = np.triu(S_k_time_Y_k.numpy())
    L_k = S_k_time_Y_k - torch.from_numpy(R_k).cpu()
    # sigma_k = Y_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1) / (S_k_list[-1].view(-1,1).transpose(0, 1) @ S_k_list[-1].view(-1,1))

    Y_k = Y_k_list[-1].view(-1, 1).double()
    S_k = S_k_list[-1].view(-1, 1).double()
    # Thực hiện phép nhân ma trận
    sigma_k = Y_k.transpose(0, 1) @ S_k / (S_k.transpose(0, 1) @ S_k)
    sigma_k=sigma_k.cpu()
    
    D_k_diag = S_k_time_Y_k.diagonal()
    upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = torch.cat([L_k.transpose(0, 1), -D_k_diag.diag()], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)
    mat_inv = mat.inverse()
    # print('mat_inv.shape',mat_inv.shape)
    v = v.view(-1,1).cpu().double()

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
    print('approx_prod',approx_prod.T, approx_prod.T.shape)    
    return approx_prod.T

def calculate_gradients(fitres_ndarrays, global_model_ndarrays, tensors_type="numpy.ndarray"):
    gradients = []
    for fitres_tensor, global_model_tensor in zip(fitres_ndarrays, global_model_ndarrays):
        # Pad the smaller tensor with zeros
        # max_size = max(fitres_tensor.size, global_model_tensor.size)
        # fitres_tensor_padded = np.pad(fitres_tensor, (0, max_size - fitres_tensor.size))
        # global_model_tensor_padded = np.pad(global_model_tensor, (0, max_size - global_model_tensor.size))
        
        # gradient = fitres_tensor_padded - global_model_tensor_padded
        gradient = fitres_tensor - global_model_tensor
        # gradients.append(gradient.tobytes())
        gradients.append(gradient)
    return Parameters(tensors=gradients, tensor_type=tensors_type)

def fld_distance(old_update_list, local_update_list, net_glob, attack_number, hvp):
    pred_update = []
    distance = []
    for i in range(len(old_update_list)):
        pred_update.append((old_update_list[i] + hvp).view(-1))
        
    
    pred_update = torch.stack(pred_update)
    local_update_list = torch.stack(local_update_list)
    old_update_list = torch.stack(old_update_list)
    #########################
    # distance = torch.norm((old_update_list - local_update_list), dim=1)
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


def detection(score, clients_state, flags, steps=1):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    
    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred

    metrics = {}
    
    if steps == 1:
        real_label=convert_clients_state_to_array(clients_state)

        accuracy = accuracy_score(real_label, label_pred)
        recall = recall_score(real_label, label_pred)
        f1 = f1_score(real_label, label_pred)
        precision = precision_score(real_label, label_pred)
        cnfs_matrix = confusion_matrix(real_label, label_pred) # confusion_matrix(y_true, y_pred).ravel()
        

        metrics["accuracy"] = accuracy
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["precision"] = precision
        metrics["confusion_matrix"] = cnfs_matrix
        # Print metrics
        log(DEBUG, "Accuracy: %s", accuracy)
        log(DEBUG, "Recall: %s", recall)
        log(DEBUG, "F1: %s", f1)
        log(DEBUG, "Precision: %s", precision)

        return label_pred, metrics

    else:
        # Second prediction
        data_pois_score = score[label_pred==1]
        data_pois_indexes = np.where(label_pred==1)[0]

        data_pois_estimator = KMeans(n_clusters=2)
        data_pois_estimator.fit(data_pois_score.reshape(-1, 1))
        
        second_label_pred = data_pois_estimator.labels_
        if np.mean(data_pois_score[second_label_pred==0])<np.mean(data_pois_score[second_label_pred==1]):
            #0 is the label of malicious clients
            second_label_pred = 1 - second_label_pred

        final_label_pred = np.copy(label_pred)
        final_label_pred[data_pois_indexes] = second_label_pred

        real_label=convert_clients_state_to_array(clients_state)
        dp_label = convert_clients_state_to_array(flags)
        
        final_label = np.logical_and(real_label, dp_label).astype(int)
        log(DEBUG, "final label: %s", final_label)
        log(DEBUG, "final label pred: %s", final_label_pred)

        # Calculate metrics
        accuracy = accuracy_score(final_label, final_label_pred)
        recall = recall_score(final_label, final_label_pred)
        f1 = f1_score(final_label, final_label_pred)
        precision = precision_score(final_label, final_label_pred)
        cnfs_matrix = confusion_matrix(final_label, final_label_pred) # confusion_matrix(y_true, y_pred).ravel()


        metrics["accuracy"] = accuracy
        metrics["recall"] = recall
        metrics["f1"] = f1
        metrics["precision"] = precision
        metrics["confusion_matrix"] = cnfs_matrix
        # Print metrics
        log(DEBUG, "Accuracy: %s", accuracy)
        log(DEBUG, "Recall: %s", recall)
        log(DEBUG, "F1: %s", f1)
        log(DEBUG, "Precision: %s", precision)


        return final_label_pred, metrics

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
    log(DEBUG, "select_k: %s", select_k)
    if select_k == 1:
        log(WARNING, 'No attack detected!')
        return 0
    else:
        log(WARNING, 'Attack Detected!')
        return 1
