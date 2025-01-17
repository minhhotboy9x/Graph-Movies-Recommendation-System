import torch
import torch.nn as nn
import torch.nn.functional as F

bce = nn.BCEWithLogitsLoss()

mse = nn.MSELoss()

rmse = lambda x, y: torch.sqrt(mse(x, y))

def calculate_bpr_loss(movie_user_edge_index, rating, pred, threshold=3.5):
    num_users = movie_user_edge_index[1].max().item() + 1
    bpr_loss = 0.0
    total_pairs = 0

    for user_id in range(num_users):
        # Lọc các phần tử thuộc user hiện tại
        mask_user = movie_user_edge_index[1] == user_id

        # Lấy positive và negative
        mask_pos = (rating >= threshold)
        mask_neg = (rating < threshold)

        pos_indices = torch.nonzero(mask_user & mask_pos).view(-1)
        neg_indices = torch.nonzero(mask_user & mask_neg).view(-1)

        if pos_indices.numel() > 0 and neg_indices.numel() > 0:
            pos_idx = pos_indices[torch.randint(0, pos_indices.size(0), (1,))].item()
            neg_idx = neg_indices[torch.randint(0, neg_indices.size(0), (1,))].item()

            r_ui = pred[pos_idx]
            r_uj = pred[neg_idx]

            # Tính loss cho cặp hiện tại
            bpr_loss += -F.logsigmoid(r_ui - r_uj)
            total_pairs += 1

    # Trả về loss trung bình
    return bpr_loss / total_pairs if total_pairs > 0 else torch.tensor(0.0)
    
