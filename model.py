import torch
import yaml
import torch.nn.functional as F
import torch_geometric.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
import utils
from dataloader import MyHeteroData

utils.set_seed(0)
class BipartiteLightGCN(nn.MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr='add')  # Aggregates neighboring nodes with 'add' method

    def forward(self, nodes: tuple[torch.Tensor, torch.Tensor], 
                edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        x, y = nodes
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

        if edge_weight is not None:
            norm = norm * edge_weight

        x2y = self.propagate(edge_index, size=(x.size(0), y.size(0)), x=(x, y), norm=norm)
        y2x = self.propagate(edge_index.flip(0), size=(y.size(0), x.size(0)), x=(y, x), norm=norm)
        return y2x, x2y

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j  # Apply normalization to the embeddings
    
    def update(self, aggr_out):
        # Takes in the output of aggregation as first argument 
        # and any argument which was initially passed to propagate()
        return aggr_out

class Classifier(torch.nn.Module):
    def forward(self, x_user: torch.Tensor, x_movie: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        edge_feat_user = x_user[edge_label_index[1]]
        edge_feat_movie = x_movie[edge_label_index[0]]

        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class HeteroLightGCN(torch.nn.Module):
    def __init__(self, hetero_metadata=None, model_config=None):
        super().__init__()
        self.model_config = model_config
        self.node_types = hetero_metadata[0]
        self.edge_types = hetero_metadata[1]
        self.exclude_node = model_config['exclude_node']

        self.embs = torch.nn.ModuleDict({
                key: torch.nn.Embedding(self.node_types[key], model_config['num_dim']) 
                    for key in self.node_types if key not in self.exclude_node
            })

        self.lightgcn = BipartiteLightGCN()
        self.classifier = Classifier()

    def forward(self, data: HeteroData):
        x_dict = {
            key: self.embs[key](data[key].node_id) 
                for key in data.node_types if key not in self.exclude_node
        }

        edge_dict = {key: data[key].edge_index 
                        for key in data.edge_types 
                            if key[0] not in self.exclude_node and key[2] not in self.exclude_node}
        
        embs_dict = {key: [x_dict[key]]
                        for key in x_dict.keys()}

        for _ in range(self.model_config['num_layers']):
            tmp_dict = {key: 0 for key in x_dict.keys()}
            for _, (key, edge_index) in enumerate(edge_dict.items()):
                edge_weight = getattr(data[key], 'weight', None)
                x = x_dict[key[0]]
                y = x_dict[key[2]]

                x, y = self.lightgcn(nodes=(x, y), edge_index=edge_index, edge_weight=edge_weight)

                tmp_dict[key[0]] = tmp_dict[key[0]] + x
                tmp_dict[key[2]] = tmp_dict[key[2]] + y

            for key in x_dict.keys():
                embs_dict[key].append(tmp_dict[key])

        res_dict = {}
        for key in embs_dict.keys():
            embs = torch.stack(embs_dict[key], dim=0)
            weights = 1.0 / (torch.arange(self.model_config['num_layers'] + 1) + 1.0)
            weights = weights.to(embs.device)
            embs = (weights.view(-1, 1, 1) * embs).sum(dim=0)
            res_dict[key] = embs

        res = self.classifier(res_dict['user'], res_dict['movie'], data['movie', 'ratedby', 'user'].edge_label_index)

        return res, res_dict

if __name__ == "__main__":
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    data = MyHeteroData(data_config)
    data.preprocess_df()
    data.create_hetero_data()
    data.split_data()
    data.create_dataloader()
    print(data.get_metadata())
    model = HeteroLightGCN(data.get_metadata(), config['model'])
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total parameters: {total_params}')
    print(f'Trainable parameters: {trainable_params}')
    for i, batch in enumerate(data.trainloader):
        print(f"Batch {i}: {batch}")
        print('-----------------')
        res, res_dict = model(batch)
        break