import torch
import yaml
import torch.nn.functional as F
import torch_geometric.nn as nn
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from dataloader import MyHeteroData

class BipartiteLightGCN(nn.MessagePassing):
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
        return self.propagate(edge_index, size=(x.size(0), y.size(0)), x=(x, y), norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j  # Apply normalization to the embeddings
    
    def update(self, aggr_out):
        # Takes in the output of aggregation as first argument 
        # and any argument which was initially passed to propagate()
        return aggr_out

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = nn.SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = nn.SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class HeteroGCN(torch.nn.Module):
    def __init__(self, num_users, num_movies, num_genre, hetero_metadata=None, model_config=None):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, model_config['num_dim'])

        self.movie_emb = torch.nn.Embedding(num_movies, model_config['num_dim'])
        self.movie_proj = nn.Linear(model_config['num_dim'] + num_genre, model_config['num_dim'])

        self.gnn = GNN(model_config['num_dim'])
        self.gnn = nn.to_hetero(self.gnn, metadata=hetero_metadata)

    def forward(self, data: MyHeteroData):
        user_feature = self.user_emb(data['user'].node_id)

        movie_feature = torch.cat([self.movie_emb(data['movie'].node_id), data['movie'].movie_genres], dim=1)
        movie_feature = self.movie_proj(movie_feature)
        
        x_dict = {'user': user_feature, 'movie': movie_feature}
        edge_index_dict = data.edge_index_dict
        print(user_feature.shape)
        print(edge_index_dict[('user', 'rates', 'movie')].shape)
        x_dict = self.gnn(x_dict, edge_index_dict)

        return x_dict

if __name__ == "__main__":
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    data = MyHeteroData(data_config)
    data.preprocess_df()
    data.create_hetero_data()
    data.split_data()
    data.create_dataloader()
    print(data.data.metadata())
    model = HeteroGCN(data.num_users, data.num_movies, data.num_genre, data.data.metadata(), config['model'])
    for i, batch in enumerate(data.trainloader):
        x_dict = model(batch)
        print(x_dict['user'].shape)
        print(x_dict['movie'].shape)
        break