import torch
import torch.nn as nn

bce = nn.BCEWithLogitsLoss()

mse = nn.MSELoss()

rmse = lambda x, y: torch.sqrt(mse(x, y))

