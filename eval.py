import os
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from dataloader import MyHeteroData
from model import HeteroLightGCN
from loss import bce
from metrics import Accuracy, F1Score
import tqdm
import utils

utils.set_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval(model, valloader):
    model.eval()
    acc_eval = Accuracy()
    f1_eval = F1Score()
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
            acc_eval.update(pred, label)
            f1_eval.update(pred, label)
            tloss = (
                (tloss * i + loss_items) / (i + 1) if tloss is not None else loss_items
            )
            pbar.set_postfix({'loss': tloss.item()})
    pbar.close()

    acc = acc_eval.compute()
    f1 = f1_eval.compute()
    
    print(f"Validation Results:")
    print(f"- Loss: {tloss.item():.4f}")
    print(f"- Accuracy: {acc:.4f}")
    print(f"- F1 Score for each class:")
    for label, f1_class in f1.items():
        print(f"    + Class {label}: {f1_class:.4f}")
    return tloss, acc
