import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import community as community_louvain
import os


class Karate_Club(InMemoryDataset):
    def __init__(self, transform=None):
        super(Karate_Club, self).__init__('.', transform, None, None)

        import networkx as nx

        home = os.path.dirname(__file__)
        self.path = os.path.join(home , 'data' , 'karate-TGF' , 'karate.txt')

        with open(self.path , "r") as file:
            lines = file.readlines()
            lines = [line.strip().split() for line in lines]
            edges = [[int(line[0]) , int(line[1])] for line in lines[1:]]
        G = nx.Graph()
        G.add_edges_from(edges)

        x = torch.eye(G.number_of_nodes(), dtype=torch.float)

        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        # Create communities.

        y = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 3., 3., 3., 1., 3., 0., 0., 2.,
        1., 2., 2., 1., 3., 1., 2., 2., 2., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.long)

        # Select a single training node for each community
        # (we just use the first one).
        train_mask = torch.zeros(y.size(0), dtype=torch.bool)
        for i in range(int(y.max()) + 1):
            train_mask[(y == i).nonzero(as_tuple=False)[0]] = True
            # train_mask[(y == i).nonzero(as_tuple=False)[1]] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

        self.data, self.slices = self.collate([data])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class Les_miserable(InMemoryDataset):
    def __init__(self, transform=None):
        super(Les_miserable, self).__init__('.', transform, None, None)

        import networkx as nx

        home = os.path.dirname(__file__)
        self.path = os.path.join(home , 'data' , 'lesmis-TGF' , 'lesmis.txt')

        with open(self.path , "r") as file:
            lines = file.readlines()
            lines = [line.strip().split() for line in lines]
            edges = [[int(line[0]) , int(line[1])] for line in lines[1:]]
        G = nx.Graph()
        G.add_edges_from(edges)

        x = torch.eye(G.number_of_nodes(), dtype=torch.float)

        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        y = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 2., 3., 2., 2., 2., 3., 4.,
        4., 4., 4., 2., 2., 3., 2., 4., 2., 2., 2., 2., 2., 4., 2., 6., 1., 1.,
        6., 6., 6., 4., 4., 4., 4., 4., 3., 3., 3., 3., 3., 3., 3., 3., 4., 4.,
        1., 1., 4., 4., 1., 2., 5., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6., 6.,
        1., 1., 1., 6., 5.], dtype=torch.long)

        # Select a single training node for each community
        # (we just use the first one).
        train_mask = torch.zeros(y.size(0), dtype=torch.bool)
        for i in range(int(y.max()) + 1):
            train_mask[(y == i).nonzero(as_tuple=False)[0]] = True
            train_mask[(y == i).nonzero(as_tuple=False)[1]] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

        self.data, self.slices = self.collate([data])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class Footbal_Club(InMemoryDataset):
    def __init__(self, transform=None):
        super(Footbal_Club, self).__init__('.', transform, None, None)

        import networkx as nx

        home = os.path.dirname(__file__)
        self.path = os.path.join(home , 'data' , 'Football-TGF' , 'football.txt')

        with open(self.path , "r") as file:
            lines = file.readlines()
            lines = [line.strip().split() for line in lines]
            edges = [[int(line[0]) , int(line[1])] for line in lines[1:]]
        G = nx.Graph()
        G.add_edges_from(edges)

        x = torch.eye(G.number_of_nodes(), dtype=torch.float)

        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        # Create communities.
        y = torch.tensor([0., 5., 0., 2., 0., 0., 0., 3., 0., 4., 0., 0., 0., 5., 5., 5., 5., 4.,
        5., 2., 5., 5., 5., 6., 5., 5., 7., 7., 7., 8., 2., 9., 1., 6., 7., 7.,
        4., 4., 4., 9., 3., 4., 4., 3., 4., 6., 2., 2., 2., 3., 2., 2., 2., 2.,
        5., 5., 5., 4., 4., 3., 6., 1., 1., 8., 9., 2., 2., 2., 2., 8., 9., 3.,
        3., 3., 3., 1., 9., 7., 2., 6., 6., 8., 6., 6., 6., 8., 9., 8., 8., 8.,
        8., 7., 8., 8., 9., 9., 1., 9., 9., 9., 7., 1., 9., 6., 2., 6., 1., 2.,
        6., 1., 1., 7., 7., 9., 9.], dtype=torch.long)

        # Select a single training node for each community
        # (we just use the first one).
        train_mask = torch.zeros(y.size(0), dtype=torch.bool)
        for i in range(int(y.max()) + 1):
            train_mask[(y == i).nonzero(as_tuple=False)[0]] = True
            train_mask[(y == i).nonzero(as_tuple=False)[1]] = True
            train_mask[(y == i).nonzero(as_tuple=False)[2]] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

        self.data, self.slices = self.collate([data])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class Dolphins(InMemoryDataset):
    def __init__(self, transform=None):
        super(Dolphins, self).__init__('.', transform, None, None)

        import networkx as nx

        home = os.path.dirname(__file__)
        self.path = os.path.join(home , 'data' , 'dolphins-TGF' , 'dolphins.txt')

        with open(self.path , "r") as file:
            lines = file.readlines()
            lines = [line.strip().split() for line in lines]
            edges = [[int(line[0]) , int(line[1])] for line in lines[1:]]
        G = nx.Graph()
        G.add_edges_from(edges)

        x = torch.eye(G.number_of_nodes(), dtype=torch.float)

        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        y = torch.tensor([0., 0., 0., 1., 0., 0., 2., 1., 2., 3., 2., 1., 2., 2., 2., 2., 2., 2.,
        2., 1., 1., 1., 3., 0., 3., 0., 2., 3., 4., 4., 4., 4., 4., 4., 4., 4.,
        4., 4., 0., 4., 4., 4., 3., 1., 3., 2., 2., 2., 1., 1., 1., 1., 4., 4.,
        4., 4., 4., 1., 2., 2., 2., 2.], dtype=torch.long)

        # Select a single training node for each community
        # (we just use the first one).
        train_mask = torch.zeros(y.size(0), dtype=torch.bool)
        for i in range(int(y.max()) + 1):
            train_mask[(y == i).nonzero(as_tuple=False)[0]] = True
            # train_mask[(y == i).nonzero(as_tuple=False)[1]] = True
            # train_mask[(y == i).nonzero(as_tuple=False)[2]] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

        self.data, self.slices = self.collate([data])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)