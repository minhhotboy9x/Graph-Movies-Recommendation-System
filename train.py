import torch

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from dataloader import MyHeteroData
from model import HeteroLightGCN
import utils

utils.set_seed(0)

def load_myheterodata(data_config):
    dataset = MyHeteroData(data_config)
    dataset.preprocess_df()
    dataset.create_hetero_data()
    dataset.split_data()
    dataset.create_dataloader()
    return dataset

def init(config_dir = None):
    # load config
    if config_dir is None:
        config_dir = 'config.yaml'
    config = utils.load_config('config.yaml')
    
    # load data
    dataset = load_myheterodata(config['data'])

    # load model
    model = HeteroLightGCN(dataset.get_metadata(), config['model'])

    optimizer, scheduler, scaler = utils.create_optimizer_scheduler_scaler(config, model)
    return config, dataset, model, optimizer, scheduler, scaler

def train_step(model, dataloader, optimizer, scheduler, scaler):
    for batch in dataloader:
        optimizer.zero_grad()



def train(config_dir = None):
    config, dataset, model, optimizer, scheduler, scaler = init(config_dir)


if __name__ == "__main__":
    train()