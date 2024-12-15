import os
import torch
import yaml
import random
import numpy as np
import importlib


# Load a YAML configuration file
def load_config(path):
    with open(path) as f:
        config_yaml = yaml.safe_load(f)
    return config_yaml



# Import an object from a string path
def import_object(path):
    module_path, obj_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


# Get the optimizer, scheduler, and scaler
def create_optimizer_scheduler_scaler(config_yaml, model):
    training_config = config_yaml["train"]
    # Optimizer
    optimizer_class = import_object(training_config["optimizer"]["type"])
    optimizer_params = training_config["optimizer"]["params"]
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    
    # Scheduler
    scheduler = None  # Default value if no scheduler is provided
    if "scheduler" in training_config:  # Check if scheduler is in config
        scheduler_class = import_object(training_config["scheduler"]["type"])
        scheduler_params = training_config["scheduler"]["params"]
        scheduler = scheduler_class(optimizer, **scheduler_params)
    
    # AMP (Automatic Mixed Precision)
    use_amp = training_config.get("amp", False)  # Default to False if not specified
    if use_amp:
        scaler = torch.amp.GradScaler()  # Create GradScaler for AMP
    else:
        scaler = None  # No AMP, no scaler
    
    return optimizer, scheduler, scaler

def set_seed(seed = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    config_yaml = load_config("config.yaml")
    # print(config)
    model = torch.nn.Linear(10, 1)
    optimizer, scheduler, scaler = create_optimizer_scheduler_scaler(config_yaml, model)
    print(optimizer)
    print(scheduler)
    print(scaler)