import argparse
import datetime
import os

import pytz
import torch
import utils
from dataloader import MyHeteroData
from evaluation import train_eval
from loss import calculate_bpr_loss, mse, rmse
from model import HeteroLightGCN
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
__current_time__ = datetime.datetime.now(vietnam_tz).strftime("%Y-%m-%d_%H-%M-%S")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# set seed for reproducibility
utils.set_seed(0)


def load_myheterodata(data_config):
    """
    Tải và xử lý dữ liệu MyHeteroData theo cấu hình dữ liệu.
    Tham số:
    data_config (dict): Cấu hình dữ liệu để tải và xử lý.
    Trả về:
    MyHeteroData: Đối tượng MyHeteroData đã được tải và xử lý.
    """

    dataset = MyHeteroData(data_config)
    dataset.preprocess_df()
    dataset.create_hetero_data()
    dataset.split_data()
    dataset.create_dataloader()
    return dataset


def init(config_dir=None):
    """
    Khởi tạo mô hình và các thành phần liên quan từ file cấu hình.
    Args:
        config_dir (str, optional): Đường dẫn tới file cấu hình. Mặc định là "config.yaml".
    Returns:
        tuple: Bao gồm các thành phần sau:
            - config (dict): Cấu hình đã được tải.
            - dataset (Dataset): Dữ liệu đã được tải.
            - model (HeteroLightGCN): Mô hình đã được khởi tạo.
            - optimizer (Optimizer): Bộ tối ưu hóa cho mô hình.
            - scheduler (Scheduler): Bộ lập lịch cho quá trình huấn luyện.
            - scaler (Scaler): Bộ scaler cho quá trình huấn luyện.
            - writer (SummaryWriter): Writer để ghi log trong quá trình huấn luyện.
            - end_epoch (int): Số lượng epoch để huấn luyện.
            - rank_k (int): Giá trị k cho đánh giá rank@k.
    """

    # load config
    if config_dir is None:
        config_dir = "config.yaml"
    config = utils.load_config("config.yaml")

    # load data
    dataset = load_myheterodata(config["data"])

    # load model
    model = HeteroLightGCN(dataset.get_metadata(), config["model"])

    optimizer, scheduler, scaler = utils.create_optimizer_scheduler_scaler(config, model)

    os.makedirs(config["logdir"], exist_ok=True)
    # num_train = len(os.listdir(config["logdir"]))
    # writer = SummaryWriter(log_dir=os.path.join(config["logdir"], f"train_{num_train}"))
    writer = SummaryWriter(log_dir=os.path.join(config["logdir"], f"train_{__current_time__}"))
    end_epoch = config["train"]["epochs"]
    rank_k = config["train"]["rank@k"]
    return config, dataset, model, optimizer, scheduler, scaler, writer, end_epoch, rank_k


def init_from_checkpoint(checkpoint):
    """
    Khởi tạo từ checkpoint đã lưu.
    Tham số:
    checkpoint (str): Đường dẫn tới file checkpoint.
    Trả về:
    tuple: Bao gồm các thành phần sau:
        - config (dict): Cấu hình của mô hình.
        - dataset: Dữ liệu đã tải.
        - model: Mô hình đã lưu.
        - optimizer: Trình tối ưu đã lưu.
        - scheduler: Bộ lập lịch đã lưu.
        - scaler: Bộ scaler đã lưu.
        - writer (SummaryWriter): Đối tượng SummaryWriter để ghi log.
        - epoch (int): Số epoch hiện tại.
        - end_epoch (int): Số epoch kết thúc.
        - train_losses (list): Danh sách các giá trị loss trong quá trình huấn luyện.
        - val_losses (list): Danh sách các giá trị loss trong quá trình xác thực.
        - rank_k: Giá trị rank@k đã lưu.
    """

    ckpt = utils.load_checkpoint(checkpoint)
    config = ckpt["config"]
    dataset = load_myheterodata(config["data"])
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
    return (
        config,
        dataset,
        model,
        optimizer,
        scheduler,
        scaler,
        writer,
        epoch,
        end_epoch,
        train_losses,
        val_losses,
        rank_k,
    )


def train_step(model, trainloader, optimizer, scheduler, scaler, threshold=4.0):
    """
    Thực hiện một bước huấn luyện cho mô hình.
    Tham số:
    - model: Mô hình cần huấn luyện.
    - trainloader: Bộ tải dữ liệu huấn luyện.
    - optimizer: Bộ tối ưu hóa để cập nhật trọng số mô hình.
    - scheduler: Bộ lập lịch để điều chỉnh tốc độ học.
    - scaler: Bộ chia tỷ lệ để hỗ trợ huấn luyện với độ chính xác hỗn hợp.
    - threshold: Ngưỡng để tính toán tổn thất BPR (mặc định là 4.0).
    Trả về:
    - t_rmse_loss: Tổn thất RMSE trung bình sau một bước huấn luyện.
    - t_bpr_loss: Tổn thất BPR trung bình sau một bước huấn luyện.
    """

    pbar = tqdm(
        enumerate(trainloader),
        desc="Training",
        total=len(trainloader),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )
    t_rmse_loss = None  # total rmse loss
    t_bpr_loss = None  # total bpr loss
    for i, batch in pbar:
        # print(f"Batch {i}: {batch['movie', 'ratedby', 'user'].edge_label}")
        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=scaler is not None):
            batch.to(device)

            edge_index = batch["movie", "ratedby", "user"].edge_index
            edge_label_index = batch["movie", "ratedby", "user"].edge_label_index
            edge_label = batch["movie", "ratedby", "user"].edge_label
            message_passing_label = batch["movie", "ratedby", "user"].rating
            edge_pred, message_passing_pred, res_dict = model(batch)

            bpr_loss = calculate_bpr_loss(
                torch.concat([edge_index, edge_label_index], dim=-1),
                torch.concat([edge_label, message_passing_label], dim=-1),
                torch.concat([edge_pred, message_passing_pred], dim=-1),
                threshold=threshold,
            )

            # NOTE: Uncomment this to use combined losses
            # Including MSE of edge prediction and message passing prediction, BPR loss
            # MSE of edge prediction and message passing prediction for quality of prediction
            # BPR loss for quality of ranking

            ## Uncomment this to use combined losses
            # loss_backprop = (
            #     mse(edge_pred, edge_label)
            #     + mse(message_passing_pred, message_passing_pred)
            #     + bpr_loss
            # )

            ## Uncomment this to use only RMSE loss
            # loss_backprop = mse(edge_pred, edge_label) + mse(
            #     message_passing_pred, message_passing_pred
            # )

            ## Uncomment this to use only BPR loss
            loss_backprop = bpr_loss

            rmse_loss = rmse(edge_pred, edge_label).detach()

            t_rmse_loss = (
                (t_rmse_loss * i + rmse_loss) / (i + 1) if t_rmse_loss is not None else rmse_loss
            )

            t_bpr_loss = (
                (t_bpr_loss * i + bpr_loss) / (i + 1) if t_bpr_loss is not None else bpr_loss
            )

        if scaler is not None:
            scaler.scale(loss_backprop).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_backprop.backward()
            optimizer.step()

        pbar.set_postfix(
            {
                f"Batch": i,
                f"RMSE loss": f"{t_rmse_loss.item():.5f}",
                f"BPR loss": f"{t_bpr_loss.item():.5f}",
            }
        )

    return t_rmse_loss, t_bpr_loss


def train(args):
    """
    Hàm train thực hiện quá trình huấn luyện mô hình.
    Tham số:
    args (Namespace): Đối tượng chứa các tham số và tùy chọn cho quá trình huấn luyện.
    Các biến khởi tạo:
    - start_epoch (int): Epoch bắt đầu huấn luyện.
    - config (dict): Cấu hình của mô hình và quá trình huấn luyện.
    - dataset (Dataset): Bộ dữ liệu huấn luyện và kiểm tra.
    - model (nn.Module): Mô hình học sâu.
    - optimizer (Optimizer): Bộ tối ưu hóa cho quá trình huấn luyện.
    - scheduler (Scheduler): Bộ điều chỉnh learning rate.
    - scaler (GradScaler): Bộ scaler cho mixed precision training.
    - writer (SummaryWriter): Đối tượng ghi log cho TensorBoard.
    - rank_k (int): Giá trị k cho các metric đánh giá.
    - epoch (int): Epoch hiện tại.
    - end_epoch (int): Epoch kết thúc huấn luyện.
    - train_losses (list): Danh sách lưu trữ các giá trị loss của tập huấn luyện.
    - val_losses (list): Danh sách lưu trữ các giá trị loss của tập kiểm tra.
    Các bước thực hiện:
    1. Khởi tạo các biến và cấu hình từ checkpoint nếu có.
    2. Nếu không resume từ checkpoint, khởi tạo optimizer, scheduler, scaler và writer.
    3. Nếu không có checkpoint, khởi tạo từ đầu với cấu hình được cung cấp.
    4. Đưa mô hình lên thiết bị (GPU/CPU).
    5. Bắt đầu quá trình huấn luyện qua các epoch.
    6. Trong mỗi epoch, thực hiện bước huấn luyện và đánh giá.
    7. Lưu checkpoint và cập nhật mô hình tốt nhất nếu cần.
    8. Lưu biểu đồ loss và ghi log các metric lên TensorBoard.
    Ngoại lệ:
    - ValueError: Nếu không cung cấp checkpoint khi resume huấn luyện.
    """

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
        (
            config,
            dataset,
            model,
            optimizer,
            scheduler,
            scaler,
            writer,
            epoch,
            end_epoch,
            train_losses,
            val_losses,
            rank_k,
        ) = init_from_checkpoint(args.checkpoint)

    if args.resume is False and args.checkpoint is not None:
        optimizer, scheduler, scaler = utils.create_optimizer_scheduler_scaler(config, model)
        # writer = SummaryWriter(
        #     log_dir=os.path.join(config["logdir"], f"train_{len(os.listdir(config['logdir']))}")
        # )
        writer = SummaryWriter(log_dir=os.path.join(config["logdir"], f"train_{__current_time__}"))
        epoch = 0
        end_epoch = config["train"]["epochs"]
        rank_k = config["train"]["rank@k"]
        train_losses = []
        val_losses = []
    elif args.checkpoint is None:
        config, dataset, model, optimizer, scheduler, scaler, writer, end_epoch, rank_k = init(
            args.config
        )
    print(model)
    if args.resume:
        if args.checkpoint is None:
            raise ValueError("Please provide a checkpoint file to resume training from.")
        print("Resuming training...")
        start_epoch = epoch + 1

    model.to(device)
    best_val_loss = float("inf")
    last_model_path = os.path.join(writer.log_dir, "last.pt")
    best_model_path = os.path.join(writer.log_dir, "best.pt")
    loss_plot_path = os.path.join(writer.log_dir, "loss_plot.png")

    print("Start training...")
    for epoch in range(start_epoch, end_epoch):

        print(f"Epoch {epoch+1}/{end_epoch}")
        t_rmse_loss, t_bpr_loss = train_step(
            model,
            dataset.trainloader,
            optimizer,
            scheduler,
            scaler,
            threshold=dataset.data_config["pos_threshold"],
        )
        scheduler.step()
        val_rmse_loss, f1_k, precision_k, recall_k, nDCG_k = train_eval(
            model, dataset.valloader, rank_k=rank_k, threshold=dataset.data_config["pos_threshold"]
        )

        train_losses.append(t_rmse_loss.detach().cpu().numpy())
        val_losses.append(val_rmse_loss.detach().cpu().numpy())

        # save model
        utils.save_checkpoint2(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            end_epoch,
            t_rmse_loss,
            val_rmse_loss,
            rank_k,
            f1_k,
            precision_k,
            recall_k,
            nDCG_k,
            last_model_path,
            config,
            writer.log_dir,
            train_losses,
            val_losses,
        )
        if val_rmse_loss < best_val_loss:
            best_val_loss = val_rmse_loss
            utils.save_checkpoint2(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                end_epoch,
                t_rmse_loss,
                val_rmse_loss,
                rank_k,
                f1_k,
                precision_k,
                recall_k,
                nDCG_k,
                best_model_path,
                config,
                writer.log_dir,
                train_losses,
                val_losses,
            )
        # save loss plot
        utils.save_loss_plot(train_losses, val_losses, loss_plot_path)
        # Log train/val metrics to TensorBoard
        writer.add_scalar("Train/RMSE loss", t_rmse_loss, epoch)
        writer.add_scalar("Train/BPR loss", t_bpr_loss, epoch)
        writer.add_scalar("Validation/Loss", val_rmse_loss, epoch)

        print("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--resume", type=bool, default=False, help="Resume training")
    args = parser.parse_args()

    train(args)
