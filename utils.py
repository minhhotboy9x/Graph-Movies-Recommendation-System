import os
import torch
import yaml
import random
import numpy as np
import importlib
import matplotlib.pyplot as plt


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

def remove_label_edges(batch):
    # print(type(output))
    movie_user_edge = batch["movie", "ratedby", "user"]
    edge_index = movie_user_edge.edge_index
    edge_label_index = movie_user_edge.edge_label_index

    edge_label_set = edge_label_index.t().unsqueeze(1)  # Shape: [n, 1, 2]
    edges = edge_index.t().unsqueeze(0)                # Shape: [1, m, 2]
    
    # So khớp tất cả cạnh trong edge_index với edge_label_index
    matches = (edges == edge_label_set).all(dim=2)     # Shape: [n, m]
    
    # Kiểm tra xem mỗi cạnh trong edge_index có khớp với ít nhất một cạnh trong edge_label_index
    mask = ~matches.any(dim=0)  # Shape: [m], True nếu cạnh không khớp
    movie_user_edge.edge_index = edge_index[:, mask]
    movie_user_edge.pos = movie_user_edge.pos[mask]
    movie_user_edge.rating = movie_user_edge.rating[mask]
    movie_user_edge.weight = movie_user_edge.weight[mask]
    movie_user_edge.e_id = movie_user_edge.e_id[mask]
    return batch

def set_seed(seed = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

def save_checkpoint(model, optimizer, scheduler, 
                    scaler, epoch, train_loss, 
                    val_loss, val_acc, val_f1, 
                    model_path, config):
    """
    Lưu toàn bộ mô hình và các thông số huấn luyện vào file checkpoint.
    
    Args:
        model: Mô hình PyTorch (hoặc bất kỳ mô hình nào).
        optimizer: Optimizer (Ví dụ: Adam, SGD, ...).
        scheduler: Scheduler (Ví dụ: Learning rate scheduler).
        scaler: Scaler dùng cho mixed precision training.
        epoch: Số epoch hiện tại.
        train_loss: Giá trị train loss.
        val_loss: Giá trị validation loss.
        val_acc: Accuracy trên validation.
        val_f1: F1 score trên validation.
        model_path: Đường dẫn đến file lưu checkpoint.
        config: Cấu hình huấn luyện.
    """
    checkpoint = {
        'model': model,              # Thông số mô hình
        'optimizer': optimizer,      # Thông số optimizer
        'scheduler': scheduler,      # Thông số scheduler
        'scaler': scaler,            # Thông số scaler (nếu có)
        'epoch': epoch,                          # Số epoch hiện tại
        'train_loss': train_loss,                # Giá trị train loss
        'val_loss': val_loss,                    # Giá trị validation loss
        'val_acc': val_acc,                      # Accuracy trên validation
        'val_f1': val_f1,                        # F1 score trên validation
        'config': config                        # Cấu hình huấn luyện
    }
    torch.save(checkpoint, model_path)

def load_checkpoint(save_path):
    """
    Load checkpoint từ file.
    
    Args:
        save_path: Đường dẫn đến file checkpoint.
    
    Returns:
        checkpoint: Dữ liệu trong checkpoint.
    """
    checkpoint = torch.load(save_path)
    return checkpoint

def save_loss_plot(train_losses, val_losses, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss', color='blue')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.savefig(file_path)
    plt.close()

if __name__ == "__main__":
    config_yaml = load_config("config.yaml")
    # print(config)
    model = torch.nn.Linear(10, 1)
    optimizer, scheduler, scaler = create_optimizer_scheduler_scaler(config_yaml, model)
    print(optimizer)
    print(scheduler)
    print(scaler)