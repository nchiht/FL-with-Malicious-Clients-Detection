# from flwr.server.strategy import FedAvg
# from flwr.common.logger import log
# from logging import DEBUG, INFO
# import numpy as np

# class CustomFedAvg(FedAvg):
#     def __init__(self, num_malicious, warmup_rounds, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.num_malicious = num_malicious
#         self.warmup_rounds = warmup_rounds
#         self.malicious_lst = []

# def configure_fit(self, server_round, parameters, client_manager):
#     # Lấy danh sách client và fit instructions từ hàm cha
#     instructions = super().configure_fit(server_round, parameters, client_manager)

#     # Randomly select malicious clients
#     if server_round == (self.warmup_rounds + 1):
#         size = self.num_malicious
#         log(INFO, "Selecting %s malicious clients", size)
#         self.malicious_lst = np.random.choice(
#             [proxy.cid for proxy, _ in instructions], size=size, replace=False
#         )

#     # Create and log clients_state
#     clients_state = dict()
#     for proxy, _ in instructions:
#         clients_state[proxy.cid] = proxy.cid in self.malicious_lst
#     clients_state = {k: clients_state[k] for k in sorted(clients_state)}  # Sort if needed
#     log(
#         DEBUG,
#         "fit_round %s: malicious clients selected %s, clients_state %s",
#         server_round,
#         self.malicious_lst,
#         clients_state,
#     )

#     # Add malicious/benign state to fit instructions
#     for proxy, fit_ins in instructions:
#         fit_ins.config["is_malicious"] = clients_state[proxy.cid]

#     return instructions