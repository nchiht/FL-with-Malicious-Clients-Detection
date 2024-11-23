import flwr
from flwr.simulation import run_simulation, start_simulation
from flwr.server import ServerApp
from flwr.client import ClientApp

from server import server_fn, EnhancedServer
from clients import client_fn, FlowerClient
import torch
from datasets.utils.logging import disable_progress_bar

from strategies import strategy


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

NUM_CLIENTS = 10
BATCH_SIZE = 32
# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`

if __name__ == '__main__':
    # main()
    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)

    # Create the ServerApp
    server = ServerApp(server_fn=server_fn)

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
        
    )
    # history = start_simulation(
    #     client_fn=client_fn,
    #     num_clients=NUM_CLIENTS,
    #     # client_resources=cfg.client_resources, # TODO: Add again when we manually set the resources
    #     #ray_init_args={"num_cpus": 1, "num_gpus": 16},
    #     server=EnhancedServer(strategy=strategy),
    #     # config=flwr.server.ServerConfig(num_rounds=cfg.server.num_rounds),
    #     strategy=strategy,
    # )