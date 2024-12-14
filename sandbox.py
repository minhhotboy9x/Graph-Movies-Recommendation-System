import yaml
import importlib

import torch
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import degree
from torch_geometric.nn import SAGEConv, MessagePassing, to_hetero, HeteroConv

with open('config.yaml') as f:
    config = yaml.safe_load(f)

class BipartiteGraphOperator(MessagePassing):
    def __init__(self):
        super(BipartiteGraphOperator, self).__init__('add')

    def forward(self, x, assign_index, N, M):
        return self.propagate(assign_index, size=(N, M), x=x)

class BipartiteLightGCN(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr='add')  # Aggregates neighboring nodes with 'add' method

    def forward(self, x, y, edge_index):
        from_, to_ = edge_index
        # Degree calculation for both sets X and Y
        deg_x = degree(from_, x.size(0), dtype=x.dtype)  # Degree for X
        deg_y = degree(to_, y.size(0), dtype=y.dtype)    # Degree for Y
        
        # Normalize degrees
        deg_x_inv_sqrt = deg_x.pow(-0.5)
        deg_y_inv_sqrt = deg_y.pow(-0.5)
        
        # Set degrees to 0 where necessary (handle divisions by 0)
        deg_x_inv_sqrt[deg_x_inv_sqrt == float('inf')] = 0
        deg_y_inv_sqrt[deg_y_inv_sqrt == float('inf')] = 0
        
        # Compute normalization factors
        norm_x = deg_x_inv_sqrt[from_]
        norm_y = deg_y_inv_sqrt[to_]
        
        # Normalize the messages with respect to both sets
        norm = norm_x * norm_y
        x2y = self.propagate(edge_index, size=(x.size(0), y.size(0)), x=(x, y), norm=norm)
        y2x = self.propagate(edge_index.flip(0), size=(y.size(0), x.size(0)), x=(y, x), norm=norm)
        return y2x, x2y

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j  # Apply normalization to the embeddings
    
    def update(self, aggr_out):
        # Takes in the output of aggregation as first argument 
        # and any argument which was initially passed to propagate()
        return aggr_out
    
conv = BipartiteLightGCN()

# x = torch.Tensor([[1, 1], [2, 2], [3, 3]])
# y = torch.Tensor([[4, 4], [5, 5], [6, 6], [7, 7]])
# x = torch.Tensor([[1, 1], [1, 1], [1, 1]])
# y = torch.Tensor([[1, 1], [1, 1], [1, 1], [1, 1]])
# row = torch.tensor([0, 1, 2, 0, 1, 2])
# col = torch.tensor([0, 0, 0, 1, 1, 3])
# index = torch.stack([row, col], dim=0)

# print(f'x shape: {x.shape}')
# print(f'y shape: {y.shape}')
# out = conv(x, y, index)

# print(out)