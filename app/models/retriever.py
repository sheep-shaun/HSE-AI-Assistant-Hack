import numpy as np
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors


class Retriever:
    def __init__(self, train_embeddings, test_embeddings, train_df, n_neighbors: int = 1):
        self.train_embeddings = train_embeddings
        self.test_embeddings = test_embeddings
        self.train_df = train_df

        self.n_neighbors = n_neighbors

        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', metric=cosine).fit(self.train_embeddings)

    @staticmethod
    def read_embeddings(path):
        with open(path, 'rb') as f:
            array = np.load(f)

        return array

    def query(self, idx):
        query_emb = np.expand_dims(self.test_embeddings[idx], axis=0)
        _, indices =  self.nbrs.kneighbors(query_emb)
        # print(indices, _)
        return self.train_df.iloc[indices[0]].author_comment.values[0]



# retriever = Retriever(
#     train_embeddings=Retriever.read_embeddings('/home/jovyan/novitskiy/HSE-AI-Assistant-Hack/data/train/train.npy'), 
#     test_embeddings=Retriever.read_embeddings('/home/jovyan/novitskiy/HSE-AI-Assistant-Hack/data/test/test.npy'), 
#     train_df=train
# )