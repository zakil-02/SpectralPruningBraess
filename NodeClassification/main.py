import warnings
warnings.filterwarnings('ignore')
import argparse
import sys
import os

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch_geometric.datasets import TUDataset, Planetoid,WebKB,Actor, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from torch_geometric.utils import to_networkx,from_networkx,homophily
from torch_geometric.transforms import RandomNodeSplit
from model import GCN
import methods
from rewiring import *
from rewiring.spectral_utils import spectral_gap
from dataloader import *
from nodeli import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import random
import pickle
import time
import csv


######### Hyperparams to use #############
#Cora --> Dropout = 0.4130296 ; LR = 0.01 ; Hidden_Dimension = 32
#Citeseer --> Dropout = 0.3130296 ; LR = 0.01 ; Hidden_Dimension = 32
#Pubmed --> Dropout = 0.4130296 ; LR = 0.01 ; Hidden_Dimension = 32
# Cornell = 0.4130296,0.001, 128
# Wisconsin = 0.5130296, 0.001,128
# Texas = 0.4130296,0.001,128
# Actor = 0.2130296,0.01,128
# ChameleonFiltered = 0.2130296,0.01,128
# ChameleonFilteredDirected = 0.4130296,0.01,128
# SquirrelFiltered = 0.5130296,0.01,128
# SquirrelFilteredDirected = 0.2130296,0.01,128
########################################

parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
parser.add_argument('--method', type=str, help='Max/Min/Add/Delete/FoSR/SDRF')
parser.add_argument('--dataset', type=str, help='Dataset to download')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in GCN')
parser.add_argument('--existing_graph', type=str,default=None, help='.pt file')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--max_iters', type=int, default=10, help='maximum number of edge change iterations')
#parser.add_argument('--removal_bound', type=float, default=0.95, help='removal bound for SDRF')
#parser.add_argument('--tau', type=int, default=163, help='Temperature for SDRF')
parser.add_argument('--update_period', type=int, default=1, help='Times to recalculate criterion')
parser.add_argument('--dropout', type=float, default=0.3130296, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--LR', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--device',type=str,default='cpu')
args = parser.parse_args()




device = torch.device(args.device)
filename = args.out
graphfile = args.existing_graph
max_iterations = args.max_iters
update_period = args.update_period
fgap = None
data_modifying = None
p = args.dropout
lr = args.LR
hidden_channel = args.hidden_dimension
avg_testacc = []
avg_acc_testallsplits = []
trainacclist = []
trainallsplits = []

planetoid_val_seeds =   [3164711608, 3255599240, 894959334,  493067366,  3349051410,511641138,  2487307261, 951126382,  530590201,  17966177]
het_val_seeds = [3164711608,894959334,2487307261,3349051410,493067366]

print(f"Loading the dataset...")
if args.datasets in ['Cora','Citeseer','Pubmed']:
    data, num_classes,num_features = load_data(args.dataset)

else :
      path = 'heterophilous-graphs/data/'
      filepath = os.path.join(path, args.dataset)
      data = np.load(filepath)
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
      data = transform(data)
      num_features = data.num_features
      num_classes = data.num_classes
datasetname, _ = os.path.splitext(args.dataset)
print(data)
print()
##=========================##=========================##=========================##=========================
graph, labels = get_graph_and_labels_from_pyg_dataset(data)
print("Calculating Different Informativeness Measures...")
nodelibef = li_node(graph, labels)
edgelibef = li_edge(graph, labels)
hadjbef = h_adj(graph, labels)
print(f'node label informativeness: {nodelibef:.2f}')
print(f'adjusted homophily: {hadjbef:.2f}')
print(f'edge label informativeness: {edgelibef:.2f}')
print("=============================================================")
print()
##=========================##=========================##=========================##=========================


if os.path.exists(graphfile):
  print("Loading graph from .pt file...")
  data = torch.load(graphfile)
  nxgraph = to_networkx(data, to_undirected=True)
  print(nxgraph)
  fgap, _, _, _ = spectral_gap(nxgraph)
  print(f"FinalGap = {fgap}")
  print()

else:
  print("Graph does not exist...")
  print("Preparing the graph for modifying...")
  nxgraph = to_networkx(data, to_undirected=True)
  print(nxgraph)
  gap, _, _, _ = spectral_gap(nxgraph)
  print(f"InitialGap = {gap}")
  print()


  if args.method == 'proxydelmin':
      newdata,fgap,data_modifying = methods.proxydelmin(data, nxgraph, args.max_iters)
      data.edge_index = torch.cat([newdata.edge_index])
      torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

  elif args.method == 'proxydelmax':
      newdata,fgap,data_modifying = methods.proxydelmax(data, nxgraph, args.max_iters)
      data.edge_index = torch.cat([newdata.edge_index])
      torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")
  
  elif args.method == 'proxyaddmax':
      newdata,fgap,data_modifying = methods.proxyaddmax(data, nxgraph, args.max_iters)
      data.edge_index = torch.cat([newdata.edge_index])
      torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")
  
  elif args.method == 'proxyaddmin':
      newdata,fgap,data_modifying = methods.proxyaddmin(data, nxgraph, args.max_iters)
      data.edge_index = torch.cat([newdata.edge_index])
      torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

  elif args.method == 'fosr':
      data,fgap,data_modifying = methods.fosr(data, args.max_iters)
      torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")
  
  else :
      print()
      print("Invalid Method...")
      sys.exit()

 
##=========================##=========================##=========================##=========================
print()

print("Calculating informativeness measure after pruning...")
graphaf, labelsaf = get_graph_and_labels_from_pyg_dataset(data)
nodeliaf = li_node(graphaf, labelsaf)
edgeliaf = li_edge(graphaf, labelsaf)
hadjaf = h_adj(graphaf, labelsaf)
print(f'node label informativeness: {nodeliaf:.2f}')
print(f'adjusted homophily: {hadjaf:.2f}')
print(f'edge label informativeness: {edgeliaf:.2f}')
print("=============================================================")
print()


##========================= Split the dataset into train/test/val ====================##
print("Splitting datasets train/val/test...")
transform2 = RandomNodeSplit(split="train_rest",num_splits = 10, num_val=0.2, num_test=0.2)
data  = transform2(data)
print(data)
print()
print("Start Training...")




start_gcn = time.time()
model = GCN(num_features,num_classes,hidden_channel, num_layers=args.num_layers)
model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

print()
for split_idx in range(1,10):
      print(f"Training for index = {split_idx}")
      train_mask = data.train_mask[:,split_idx]
      test_mask = data.test_mask[:,split_idx]
      val_mask = data.val_mask[:,split_idx]

      def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index,p)  # Perform a single forward pass.
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        train_correct = pred[train_mask] == data.y[train_mask]  # Check against ground-truth labels.
        train_acc = int(train_correct.sum()) / int(train_mask.sum())  # Derive ratio of correct predictions.
        return loss, train_acc
        #return loss

      def val():
        model.eval()
        out = model(data.x, data.edge_index,p)
        pred = out.argmax(dim=1)  # Use the class with highest probability. 
        val_correct = pred[val_mask] == data.y[val_mask]  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
        return val_acc


      def test():
            model.eval()
            out= model(data.x, data.edge_index,p)
            pred = out.argmax(dim=1)  # Use the class with highest probability. 
            test_correct = pred[test_mask] == data.y[test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
            return test_acc

      if datasetname in ['Cora','CiteSeer','PubMed']:
            val_seeds = planetoid_val_seeds

      else:
            val_seeds = het_val_seeds

      for seeds in val_seeds:
                set_seed(seeds)
                print("Start training ....")
                for epoch in tqdm(range(1, 101)):
                    loss, train_acc = train()
                end_gcn = time.time()
                print()
                val_acc = val()
                test_acc = test()
                trainacclist.append(train_acc*100)
                avg_testacc.append(test_acc*100)
                print(f'Val Accuracy : {val_acc:.2f}, Test Accuracy: {test_acc:.2f} for seed',seeds)
      avg_acc_testallsplits.append(np.mean(avg_testacc))
      trainallsplits.append(np.mean(trainacclist))
print(f'Final test accuracy of all splits {np.mean(avg_acc_testallsplits):.2f} \u00B1 {np.std(avg_acc_testallsplits):.2f}')
print(f'Final train accuracy of all splits {np.mean(trainallsplits):.2f} \u00B1 {np.std(trainallsplits):.2f}')
gcn_time = (f"GCN Training Time After Adding -- {end_gcn - start_gcn}")
print()  
headers = ['Method','Dataset','NumLayers','InitialGap','NLIBefore','AdjHomBefore','EdgesModified','UpdatePeriod','FinalGap','NLIAfter','AdjHomAfter','LR','Dropout','HiddenDimension','AvgTrainingAcc','Deviation','AvgTestAcc', 'Deviation', 'ModifyingTime','GCNTrainingTime']
with open(filename, mode='a', newline='') as file:
              writer = csv.writer(file)
              if file.tell() == 0:
                      writer.writerow(headers)
              writer.writerow([args.method,args.dataset,args.num_layers,initialgap,nodelibef,hadjbef,max_iterations*update_period,update_period,fgap,nodeliaf,hadjaf,lr,p,hidden_channel,np.mean(trainallsplits),np.std(trainallsplits),np.mean(avg_acc_testallsplits), np.std(avg_acc_testallsplits), data_modifying,gcn_time])
