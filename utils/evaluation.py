from typing import Callable, Dict, Optional, Tuple, List


def update_confusion_matrix(confusion_matrix: Dict[str, int], clients_states: Dict[str, bool], malicious_clients_idx: List, good_clients_idx: List):
    """Update TN, FP, FN, TP of confusion matrix."""
    print("confusion_matrix", confusion_matrix)
    print("clients_states", clients_states)
    print("malicious_clients_idx", malicious_clients_idx)
    print("good_clients_idx", good_clients_idx)

    for client_idx, client_state in clients_states.items():
        if int(client_idx) in malicious_clients_idx:
            if client_state:
                confusion_matrix["TP"] += 1
            else:
                confusion_matrix["FP"] += 1
        elif int(client_idx) in good_clients_idx:
            if client_state:
                confusion_matrix["FN"] += 1
            else:
                confusion_matrix["TN"] += 1
    print("updated confusion_matrix", confusion_matrix)
    return confusion_matrix