import torch
import pandas as pd
import yaml

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, LinkNeighborLoader

torch.random.manual_seed(0)

# Lưu ý, các timestamp trong 
class MyHeteroData():
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.ratings = pd.read_csv(data_config['ratings_path'])
        self.movies = pd.read_csv(data_config['movies_path'])
        self.links = pd.read_csv(data_config['links_path'])
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

        self.genres = self.movies['genres'].str.get_dummies('|')
    
    def create_hetero_data(self):
        self.data["user"].node_id = torch.arange(len(self.unique_user_id))
        self.data["movie"].node_id = torch.arange(len(self.unique_movie_id))

        self.num_users = self.data["user"].num_nodes = len(self.unique_user_id)
        self.num_movies = self.data["movie"].num_nodes = len(self.unique_movie_id)

        movie_genres = torch.from_numpy(self.genres.values).to(torch.float)
        self.data["movie"].movie_genres = movie_genres
        self.num_genre = self.data["movie"].movie_genres.size(1)

        ratings_user_id = torch.from_numpy(self.ratings['mappedID_x'].values).to(torch.long)
        ratings_movie_id = torch.from_numpy(self.ratings['mappedID_y'].values).to(torch.long)
        edge_index_user_to_movie = torch.stack([ratings_user_id, ratings_movie_id], dim=0)
        self.data["user", "rates", "movie"].edge_index = edge_index_user_to_movie
        rating = torch.from_numpy(self.ratings['rating'].values).to(torch.float)
        self.data["user", "rates", "movie"].rating = rating

        self.data = T.ToUndirected()(self.data)
        del self.ratings, self.movies, self.links # free memory
        # print(self.data)
        # print(self.data.num_nodes)
        # print(self.data.edge_index_dict)
    
    def split_data(self):
        transform = T.RandomLinkSplit(
                    num_val=self.data_config["val_ratio"],
                    num_test=self.data_config["test_ratio"],
                    add_negative_train_samples=False,
                    edge_types=("user", "rates", "movie"),
                    rev_edge_types=("movie", "rev_rates", "user"),
                )
        self.train_data, self.val_data, self.test_data = transform(self.data)
        # print(self.train_data)
        # print(self.val_data)
        # print(self.test_data)
    
    def create_dataloader(self):
        batch_size = self.data_config['batch_size']
        self.trainloader = LinkNeighborLoader(
            self.train_data,
            batch_size = batch_size,
            shuffle = True,
            edge_label_index = ("user", "rates", "movie"), 
            edge_label = self.train_data["user", "rates", "movie"].edge_label,  
            num_neighbors = self.data_config['num_neighbors'],  
        )
        self.valloader = LinkNeighborLoader(
            self.val_data,
            batch_size = batch_size,
            shuffle = False,
            edge_label_index = ("user", "rates", "movie"), 
            edge_label = self.val_data["user", "rates", "movie"].edge_label,  
            num_neighbors = self.data_config['num_neighbors'],  
        )
        self.testloader = LinkNeighborLoader(
            self.test_data,
            batch_size = batch_size,
            shuffle = False,
            edge_label_index = ("user", "rates", "movie"), 
            edge_label = self.test_data["user", "rates", "movie"].edge_label,  
            num_neighbors = self.data_config['num_neighbors'],  
        )

    def load_batches(self):
        for i, batch in enumerate(self.trainloader):
            # print(f"Batch {i}:")
            # print(batch["user", "rates", "movie"].edge_index)
            # print(batch["user", "rates", "movie"].edge_label_index)
            # print(batch["user"].node_id)
            print(batch.edge_index_dict[('user', 'rates', 'movie')].shape)

            break  # chỉ xem thử batch đầu tiên

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