import os
import torch
import argparse

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
    model.to(device)
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
    return tloss, acc, f1

def load_myheterodata(data_config):
    dataset = MyHeteroData(data_config)
    dataset.preprocess_df()
    dataset.create_hetero_data()
    dataset.split_data()
    dataset.create_dataloader()
    return dataset

def init_from_checkpoint(checkpoint):
    ckpt = utils.load_checkpoint(checkpoint)
    config = ckpt["config"]
    dataset = load_myheterodata(config['data'])
    model = ckpt["model"]
    return config, dataset, model

def eval(args):
    config, dataset, model = init_from_checkpoint(args.checkpoint)
    if args.split == 'test':
        valloader = dataset.testloader
    else:
        valloader = dataset.valloader
    val_loss, val_acc, val_f1 = train_eval(model, valloader)
    return val_loss, val_acc, val_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--split", type=str, choices=['val', 'test'], default='test' ,help="Split to evaluate on.")
    args = parser.parse_args()
    
    eval(args)