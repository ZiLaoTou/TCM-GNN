import dgl
import torch
from dataset import GraphDataset
from dgl.dataloading import GraphDataLoader


def get_loader(args, train_ids, test_ids):

    graphs, graph_attr = dgl.load_graphs(args.data_path)
    labels = graph_attr['labels']
    pool_ids = graph_attr['pool_ids']
    sites = graph_attr['sites']

    train_graphs, test_graphs = [], []
    train_labels, test_labels = [], []
    train_pool_ids, test_pool_ids = [], []
    train_site_ids, test_site_ids = [], []
    for train_idx in train_ids:
        train_graphs.append(graphs[train_idx])
        train_labels.append(labels[train_idx])
        train_pool_ids.append(pool_ids[train_idx])
        train_site_ids.append(sites[train_idx])
    for test_idx in test_ids:
        test_graphs.append(graphs[test_idx])
        test_labels.append(labels[test_idx])
        test_pool_ids.append(pool_ids[test_idx])
        test_site_ids.append(sites[test_idx])
    train_labels = torch.FloatTensor(train_labels)
    test_labels = torch.FloatTensor(test_labels)
    train_pool_ids = torch.stack(train_pool_ids)
    test_pool_ids = torch.stack(test_pool_ids)
    train_site_ids = torch.tensor(train_site_ids)
    test_site_ids = torch.tensor(test_site_ids)

    train_dataset = GraphDataset('train', train_graphs, train_labels, train_pool_ids, train_site_ids)
    val_dataset = GraphDataset('val', train_graphs, train_labels, train_pool_ids, train_site_ids)
    test_dataset = GraphDataset('test', test_graphs, test_labels, test_pool_ids, test_site_ids)

    train_loader = GraphDataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers = args.workers, pin_memory=True)
    val_loader = GraphDataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, num_workers = args.workers, pin_memory=True)
    test_loader = GraphDataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers = args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader