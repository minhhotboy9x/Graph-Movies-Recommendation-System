import os
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from dataloader import MyHeteroData
from model import HeteroLightGCN
from loss import bce
from metrics import Accuracy
import tqdm
import utils

utils.set_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval(model, valloader):
    model.eval()
    accuracy = Accuracy()
    pbar = tqdm.tqdm(enumerate(valloader), desc="Validation", total=len(valloader),
                     bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    tloss = None
    for i, batch in pbar:
        with torch.no_grad():
            batch.to(device)
            label = batch['movie', 'ratedby', 'user'].edge_label
            res, res_dict = model(batch)
            loss_items = bce(res, label)

            pred = torch.sigmoid(res) >= model.model_config['threshold']
            accuracy.update(pred, label)

            tloss = (
                (tloss * i + loss_items) / (i + 1) if tloss is not None else loss_items
            )
            pbar.set_postfix({'loss': tloss.item()})
    
    acc = accuracy.compute()

    pbar.set_postfix({'loss': tloss.item(), 'accuracy': acc})
    return tloss, acc
