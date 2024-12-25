import torch
import numpy as np
import pandas as pd
import yaml
import ast

import utils
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from sklearn.preprocessing import LabelEncoder

utils.set_seed(0)

class MyHeteroData():
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.ratings = pd.read_csv(data_config['ratings_path'])
        self.movies = pd.read_csv(data_config['movies_path'])
        self.links = pd.read_csv(data_config['links_path'])
        self.production = pd.read_csv(data_config['productions_path'])
        self.data = HeteroData()
    
    def preprocess_df(self):
        # map user and movie id to a unique id
        self.unique_user_id = self.ratings['userId'].unique()
        self.unique_user_id = pd.DataFrame(data={
            'userId': self.unique_user_id,
            'mappedID': pd.RangeIndex(len(self.unique_user_id)),
        })

        self.unique_movie_id = self.movies['movieId'].unique()
        self.unique_movie_id = pd.DataFrame(data={
            'movieId': self.unique_movie_id,
            'mappedID': pd.RangeIndex(len(self.unique_movie_id)),
        })

        self.ratings = pd.merge(self.ratings, self.unique_user_id, on='userId', how='left')
        self.ratings = pd.merge(self.ratings, self.unique_movie_id, on='movieId', how='left')
        self.ratings = pd.merge(self.ratings, self.links, on='movieId', how='left')
        self.ratings = self.ratings.drop(columns=['userId', 'movieId', 'timestamp', 'imdbId', 'tmdbId'])

        self.movies = pd.merge(self.movies, self.unique_movie_id, on='movieId', how='left')

        self.production = self.production.drop(columns=['imdbId', 'imdbId'])
        self.production = pd.merge(self.production, self.unique_movie_id, on='movieId', how='left')

    def create_user_movie_edges(self):
        self.data["user"].node_id = torch.arange(len(self.unique_user_id))
        self.data["movie"].node_id = torch.arange(len(self.unique_movie_id))

        self.num_users = self.data["user"].num_nodes = len(self.unique_user_id)
        self.num_movies = self.data["movie"].num_nodes = len(self.unique_movie_id)

        ratings_user_id = torch.from_numpy(self.ratings['mappedID_x'].values).to(torch.long)
        ratings_movie_id = torch.from_numpy(self.ratings['mappedID_y'].values).to(torch.long)
        # edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
        edge_index_user_to_movie = torch.stack([ratings_movie_id, ratings_user_id], dim=0)

        self.data["movie", "ratedby", "user"].edge_index = edge_index_user_to_movie.contiguous()
        rating = torch.from_numpy(self.ratings['rating'].values).to(torch.float)
        self.data["movie", "ratedby", "user"].rating = rating
        self.data["movie", "ratedby", "user"].pos = (rating >= self.data_config['pos_threshold']).float()

        # weight user movie edges
        k = self.data_config['weight_user_movie']['k']
        c = self.data_config['weight_user_movie']['c']
        self.data["movie", "ratedby", "user"].weight = 1/(1 + torch.exp(-k * (rating - c)))
                        
        # print(self.data)

    def create_movie_genre_edges(self):
        all_genres = set(genre for genres in self.movies['genres'] for genre in genres.split('|'))

        self.data["genre"].node_id = torch.arange(len(all_genres))
        self.num_genres = self.data["genre"].num_nodes = len(all_genres)

        genre_to_id = {genre: idx for idx, genre in enumerate(all_genres)}
        
        edges = []
        for _, row in self.movies.iterrows():
            movie_id = row['mappedID']  
            genres = row['genres'].split('|')  
            for genre in genres:
                genre_id = genre_to_id[genre] 
                edges.append((genre_id, movie_id))  
        edges_array = np.array(edges, dtype=np.int64)
        self.data['genre', 'of', 'movie'].edge_index = torch.from_numpy(edges_array.T).contiguous()

    def create_movie_production_edges(self):
        cols = ['director', 'writers', 'stars']

        def create_edge(col_name):    
            self.production[col_name] = self.production[col_name].apply(ast.literal_eval)
            exploded = self.production.explode(col_name)
            # exploded = exploded[exploded[col_name].notnull()]
            le = LabelEncoder()
            exploded[col_name] = le.fit_transform(exploded[col_name])

            self.production[col_name] = (
                exploded.groupby(exploded.index)[col_name].apply(list)
            )
            edges = []
            for _, row in self.production.iterrows():
                movie_id = row['mappedID']
                objects = row[col_name]
                for obj in objects:
                    edges.append((obj, movie_id))

            edges_array = np.array(edges, dtype=np.int64)
            self.data[col_name].node_id = torch.arange(len(le.classes_))
            setattr(self, f'num_{col_name}', len(le.classes_))
            self.data[col_name].num_nodes = len(le.classes_)
            # print(col_name, self.data[col_name].num_nodes)
            self.data[col_name, 'in', 'movie'].edge_index = torch.from_numpy(edges_array.T).contiguous()

        for col in cols:
            create_edge(col)


    def create_hetero_data(self):
        self.create_user_movie_edges()
        self.create_movie_genre_edges()
        self.create_movie_production_edges()
        # self.data = T.ToUndirected()(self.data)
        for node_type in self.data.node_types:
            print(f"Number of nodes for '{node_type}': {self.data[node_type].num_nodes}")
        for edge_type in self.data.edge_types:
            print(f"Number of edges for '{edge_type}': {self.data[edge_type].edge_index.size(1)}")

        del self.ratings, self.movies, self.links, self.production
    
    def split_data(self):
        transform = T.RandomLinkSplit(
                    num_val=self.data_config["val_ratio"],
                    num_test=self.data_config["test_ratio"],
                    add_negative_train_samples=False,
                    edge_types=("movie", "ratedby", "user"),
                    disjoint_train_ratio=0.2
                    # rev_edge_types=("movie", "rev_rates", "user"),
                )
        self.train_data, self.val_data, self.test_data = transform(self.data)

    def create_dataloader(self):
        batch_size = self.data_config['batch_size']
        self.trainloader = LinkNeighborLoader(
            self.train_data,
            batch_size = batch_size,
            shuffle = True,
            edge_label_index = (("movie", "ratedby", "user"), 
                                self.train_data["movie", "ratedby", "user"].edge_label_index), 
            edge_label = self.train_data["movie", "ratedby", "user"].pos,
            num_neighbors = self.data_config['num_neighbors'], 
        )
        self.valloader = LinkNeighborLoader(
            self.val_data,
            batch_size = batch_size,
            shuffle = False,
            edge_label_index = (("movie", "ratedby", "user"), 
                                self.val_data["movie", "ratedby", "user"].edge_label_index), 
            edge_label = self.val_data["movie", "ratedby", "user"].pos,  
            num_neighbors = self.data_config['num_neighbors'],  
        )
        self.testloader = LinkNeighborLoader(
            self.test_data,
            batch_size = batch_size,
            shuffle = False,
            edge_label_index = (("movie", "ratedby", "user"), 
                                self.test_data["movie", "ratedby", "user"].edge_label_index), 
            edge_label = self.test_data["movie", "ratedby", "user"].pos,  
            num_neighbors = self.data_config['num_neighbors'],  
        )
    
    def load_batches(self):
        for i, batch in enumerate(self.trainloader):
            print('-----------------')
            edge = batch["movie", "ratedby", "user"]
            print(batch)
            # print(edge)
            # Kiểm tra xem edge_label_index có nằm trong edge_index hay không
            # edge_index = edge.edge_index  # Lấy edge_index
            # edge_label_index = edge.edge_label_index  # Lấy edge_label_index

            # # Chuyển edge_index và edge_label_index sang dạng set để so sánh
            # edge_index_set = set(map(tuple, edge_index.t().tolist()))
            # edge_label_set = set(map(tuple, edge_label_index.t().tolist()))

            # common_edges = edge_label_set.intersection(edge_index_set)
            # print(f"Number of common edges: {len(common_edges)}")
            # print(f"Common edges: {common_edges}")
            if i == 0:  
                break  

    def get_metadata(self):
        meta_tmp = self.data.metadata()
        meta_data = [{}, meta_tmp[1]]
        for key in meta_tmp[0]:
            meta_data[0][key] = self.data[key].num_nodes
        return meta_data
    
if __name__ == "__main__":
    # genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    #        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror, Musical',
    #        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']
    
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    data = MyHeteroData(data_config)
    data.preprocess_df()
    data.create_hetero_data()
    data.split_data()
    data.create_dataloader()
    data.load_batches()
    # data.get_metadata()
