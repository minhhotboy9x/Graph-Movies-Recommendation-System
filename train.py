import torch

from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from dataloader import MyHeteroData
from model import HeteroLightGCN
import tqdm
import utils

utils.set_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

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

def train_step(model, trainloader, valloader, optimizer, scheduler, scaler):
    pbar = tqdm.tqdm(enumerate(trainloader), desc="Training", total=len(trainloader),
                     bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for i, batch in pbar:
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=scaler is not None):
            batch.to(device)
            label = batch['movie', 'ratedby', 'user'].edge_label
            res, res_dict = model(batch)
            pbar.set_postfix({
                f"batch": i})


def train(config_dir = None):
    config, dataset, model, optimizer, scheduler, scaler = init(config_dir)
    model.to(device)
    writer = SummaryWriter()
    for epoch in range(1):
        train_step(model, dataset.trainloader, dataset.valloader, optimizer, scheduler, scaler)

if __name__ == "__main__":
    train()