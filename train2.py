import os
import torch
import argparse
from torch.amp import autocast
from dataloader2 import MyHeteroData
from model2 import HeteroLightGCN
from loss import mse, rmse
from eval2 import train_eval
import tqdm
import utils

utils.set_seed(0)

from torch.utils.tensorboard import SummaryWriter

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

    os.makedirs(config['logdir'], exist_ok=True)
    num_train = len(os.listdir(config['logdir']))
    writer = SummaryWriter(log_dir=os.path.join(config['logdir'], f"train_{num_train}"))
    end_epoch = config['train']['epochs']
    rank_k = config['train']['rank@k']
    return config, dataset, model, optimizer, scheduler, scaler, writer, end_epoch, rank_k

def init_from_checkpoint(checkpoint):
    ckpt = utils.load_checkpoint(checkpoint)
    config = ckpt["config"]
    dataset = load_myheterodata(config['data'])
    model = ckpt["model"]
    optimizer = ckpt["optimizer"]
    scheduler = ckpt["scheduler"]
    scaler = ckpt["scaler"]
    log_dir = ckpt["log_dir"]
    writer = SummaryWriter(log_dir=log_dir)
    epoch = ckpt["epoch"]
    end_epoch = ckpt["end_epoch"]
    train_losses = ckpt["train_losses"]
    val_losses = ckpt["val_losses"]
    rank_k = ckpt["rank@k"]
    return config, dataset, model, optimizer, scheduler, scaler, writer, \
          epoch, end_epoch, train_losses, val_losses, rank_k

def train_step(model, trainloader, optimizer, scheduler, scaler):
    pbar = tqdm.tqdm(enumerate(trainloader), desc="Training", total=len(trainloader),
                     bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    tloss = None # total loss
    for i, batch in pbar:
        # print(f"Batch {i}: {batch['movie', 'ratedby', 'user'].edge_label}")
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=scaler is not None):
            batch.to(device)
            # label = batch['movie', 'ratedby', 'user'].edge_label
            label = batch['movie', 'ratedby', 'user'].edge_label
            label2 = batch['movie', 'ratedby', 'user'].rating
            res, res2, res_dict = model(batch)
            loss_backprop = mse(res, label) + mse(res2, label2)

            loss_items = rmse(res, label).detach()

            tloss = (
                (tloss * i + loss_items) / (i + 1) if tloss is not None else loss_items
            )

        if scaler is not None:
            scaler.scale(loss_backprop).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_backprop.backward()
            optimizer.step()

        pbar.set_postfix({
            f"batch": i,
            f"train rmse loss": tloss.item()})
    
    return tloss


def train(args):
    start_epoch = 0
    config = None
    dataset = None
    model = None
    optimizer = None
    scheduler = None
    scaler = None
    writer = None
    rank_k = None
    epoch = 0
    end_epoch = 0
    train_losses = []
    val_losses = []

    if args.checkpoint is not None:
        config, dataset, model, optimizer, scheduler, scaler,\
              writer, epoch, end_epoch, train_losses, val_losses, rank_k = init_from_checkpoint(args.checkpoint)
        
    if args.resume is False and args.checkpoint is not None:
        optimizer, scheduler, scaler = utils.create_optimizer_scheduler_scaler(config, model)
        writer = SummaryWriter(log_dir=os.path.join(config['logdir'], f"train_{len(os.listdir(config['logdir']))}"))
        epoch = 0
        end_epoch = config['train']['epochs']
        rank_k = config['train']['rank@k']
        train_losses = []
        val_losses = []
    elif args.checkpoint is None:
        config, dataset, model, optimizer, scheduler, scaler, writer, end_epoch, rank_k = init(args.config)
    print(model)
    if args.resume:
        if args.checkpoint is None:
            raise ValueError("Please provide a checkpoint file to resume training from.")
        print("Resuming training...")
        start_epoch = epoch+1
        
    model.to(device)
    best_val_loss = float('inf') 
    last_model_path = os.path.join(writer.log_dir, "last.pt")
    best_model_path = os.path.join(writer.log_dir, "best.pt")
    loss_plot_path = os.path.join(writer.log_dir, "loss_plot.png")
    
    print("Start training...")
    for epoch in range(start_epoch, end_epoch):
        
        print(f"Epoch {epoch+1}/{end_epoch}")
        train_loss = train_step(model, dataset.trainloader, 
                                    optimizer, scheduler, scaler)
        scheduler.step()
        val_loss, f1_k, precision_k, recall_k, nDCG_k = train_eval(model, 
                                            dataset.valloader, rank_k=rank_k, 
                                            threshold=dataset.data_config["pos_threshold"])
        
        train_losses.append(train_loss.detach().cpu().numpy())
        val_losses.append(val_loss.detach().cpu().numpy())

        # save model
        utils.save_checkpoint2(model, optimizer, scheduler, scaler, epoch, end_epoch, train_loss,
                                val_loss, rank_k, f1_k, precision_k, recall_k,
                                nDCG_k, last_model_path, config, 
                                writer.log_dir, train_losses, val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            utils.save_checkpoint2(model, optimizer, scheduler, scaler, epoch, end_epoch, train_loss,
                                val_loss, rank_k, f1_k, precision_k, recall_k,
                                nDCG_k, best_model_path, config, 
                                writer.log_dir, train_losses, val_losses)
        # save loss plot 
        utils.save_loss_plot(train_losses, val_losses, loss_plot_path)
        # Log train/val metrics to TensorBoard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        
        print('-'*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', type=str, default="config.yaml" ,help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training')
    args = parser.parse_args()

    train(args)