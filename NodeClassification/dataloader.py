import random
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid,WebKB,Actor, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def load_data(datasetname):
        path = '../data/' + datasetname
        if datasetname in ['Cora','Citeseer','Pubmed']:
            dataset = Planetoid(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print(data)

        if datasetname in ['cornell', 'texas', 'wisconsin','actor','chameleon','squirrel']:
            dataset = np.load(dataset)
            print("Converting to PyG dataset...")
            x = torch.tensor(data['node_features'], dtype=torch.float)
            y = torch.tensor(data['node_labels'], dtype=torch.long)
            edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
            train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
            val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
            test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
            num_classes = len(torch.unique(y))
            num_targets = 1 if num_classes == 2 else num_classes
            data = Data(x=x, edge_index=edge_index)
            data.y = y
            data.num_classes = num_classes
            data.num_targets = num_targets
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
            print("Done!..")
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print(data)   


        return data, num_classes,num_features