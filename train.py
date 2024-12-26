import os
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from dataloader import MyHeteroData
from model import HeteroLightGCN
from loss import bce
from eval import train_eval
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

    os.makedirs(config['logdir'], exist_ok=True)
    num_train = len(os.listdir(config['logdir']))
    writer = SummaryWriter(log_dir=os.path.join(config['logdir'], f"train_{num_train}"))
    
    return config, dataset, model, optimizer, scheduler, scaler, writer

def train_step(model, trainloader, optimizer, scheduler, scaler):
    pbar = tqdm.tqdm(enumerate(trainloader), desc="Training", total=len(trainloader),
                     bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    tloss = None # total loss
    for i, batch in pbar:
        # print(f"Batch {i}: {batch['movie', 'ratedby', 'user'].edge_label}")
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=scaler is not None):
            batch.to(device)
            label = batch['movie', 'ratedby', 'user'].edge_label
            res, res_dict = model(batch)
            loss_items = bce(res, label)

            tloss = (
                (tloss * i + loss_items) / (i + 1) if tloss is not None else loss_items
            )

        if scaler is not None:
            scaler.scale(loss_items).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_items.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        pbar.set_postfix({
            f"batch": i,
            f"train loss": loss_items.item()})
    
    return tloss


def train(config_dir = None):
    config, dataset, model, optimizer, scheduler, scaler, writer = init(config_dir)
    model.to(device)

    best_val_loss = float('inf') 
    last_model_path = os.path.join(writer.log_dir, "last.pt")
    best_model_path = os.path.join(writer.log_dir, "best.pt")
    loss_plot_path = os.path.join(writer.log_dir, "loss_plot.png")
    
    train_losses = []
    val_losses = []
    print("Start training...")
    for epoch in range(2):
        
        print(f"Epoch {epoch}")
        train_loss = train_step(model, dataset.trainloader, 
                                    optimizer, scheduler, scaler)
        scheduler.step()
        val_loss, val_acc, val_f1 = train_eval(model, dataset.valloader)
        train_losses.append(train_loss.detach().cpu().numpy())
        val_losses.append(val_loss.detach().cpu().numpy())

        # save model
        utils.save_checkpoint(model, optimizer, scheduler, scaler, epoch, train_loss,
                                val_loss, val_acc, val_f1, last_model_path, config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            utils.save_checkpoint(model, optimizer, scheduler, scaler, epoch, train_loss,
                                val_loss, val_acc, val_f1, best_model_path, config)
        # save loss plot 
        utils.save_loss_plot(train_losses, val_losses, loss_plot_path)
        # Log train/val metrics to TensorBoard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_acc, epoch)
        
        # If f1 is a dictionary, log each class's f1 score as well
        for label, f1_class in val_f1.items():
            writer.add_scalar(f'Validation/F1/Class_{label}', f1_class, epoch)
        
        # Optionally, log average F1 score
        writer.add_scalar('Validation/F1/Average', sum(val_f1.values()) / len(val_f1), epoch)

        print('-'*50)

if __name__ == "__main__":
    train()