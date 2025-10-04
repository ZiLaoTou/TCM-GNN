from dgl.data import DGLDataset

class GraphDataset(DGLDataset):
    def __init__(self, name, graphs, labels, pool_ids, sites):
        super(GraphDataset, self).__init__(name)
        self.graphs = graphs
        self.labels = labels
        self.pool_ids = pool_ids
        self.sites = sites

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.pool_ids[idx], self.sites[idx]

    def process(self):
        pass
