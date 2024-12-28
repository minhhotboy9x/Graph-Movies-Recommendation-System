import os
import torch
import argparse

from dataloader2 import MyHeteroData
from model2 import HeteroLightGCN
from loss import mse, rmse
from metrics import Accuracy, F1Score
import tqdm
import utils

utils.set_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_eval(model, valloader, rank_k=5):
    model.eval()
    model.to(device)
    
    pbar = tqdm.tqdm(enumerate(valloader), desc="Validation", total=len(valloader),
                     bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    tloss = None
    for i, batch in pbar:
        with torch.no_grad():
            batch.to(device)
            label = batch['movie', 'ratedby', 'user'].edge_label
            res, res_dict = model(batch, mode='val')

            loss_items = rmse(res, label)
            f1_k = 0
            nDCG_k = 0
            tloss = (
                (tloss * i + loss_items) / (i + 1) if tloss is not None else loss_items
            )
            pbar.set_postfix({'loss': tloss.item()})
    pbar.close()

    print(f"Validation Results:")
    print(f"- Loss: {tloss.item():.4f}")
    print(f"- F1@{rank_k}: {f1_k:.4f}")
    print(f"- nDCG@{rank_k}: {nDCG_k:.4f}")
    return tloss, f1_k, nDCG_k

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
    rank_k = ckpt["rank@k"]
    return config, dataset, model, rank_k

def eval(args):
    config, dataset, model, rank_k = init_from_checkpoint(args.checkpoint)
    if args.split == 'test':
        valloader = dataset.testloader
    else:
        valloader = dataset.valloader
    tloss, f1_k, nDCG_k = train_eval(model, valloader, rank_k)
    return tloss, f1_k, nDCG_k, rank_k

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--split", type=str, choices=['val', 'test'], default='test' ,help="Split to evaluate on.")
    args = parser.parse_args()
    
    eval(args)