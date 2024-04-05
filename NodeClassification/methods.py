import time
import torch
import networkx as nx
from dataloader import *
from tqdm import tqdm
from rewiring.fastrewiringKupdates import *
from rewiring.MinGapKupdates import *
from rewiring.fosr import *
from rewiring.spectral_utils import *
from rewiring.sdrf import *
from torch_geometric.utils import to_networkx,from_networkx,homophily




def proxydelmin(data, nxgraph, max_iterations):
    print("Deleting edges to minimize the gap...")
    start_algo = time.time()
    newgraph = min_and_update_edges(nxgraph, rank_by_proxy_delete_min, "proxydeletemin", max_iter=max_iterations, updating_period=1)
    newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
    end_algo = time.time()
    fgap, _, _, _ = spectral_gap(newgraph)
    print()
    print(f"FinalGap = {fgap}") 
    data_modifying = (f" Modifying time = {end_algo - start_algo}")
    print(data_modifying)  
    newdata = from_networkx(newgraph)
    print(newdata)
    data.edge_index = torch.cat([newdata.edge_index]) 
    return data,fgap,data_modifying

def proxydelmax(args,data, nxgraph, max_iterations):
      print("Deleting edges to maximize the gap...")
      start_algo = time.time()
      newgraph = process_and_update_edges(nxgraph, rank_by_proxy_delete, "proxydeletemax",max_iter=max_iterations,updating_period=1)
      newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
      end_algo = time.time()
      fgap,_, _, _ = spectral_gap(newgraph)
      print()
      print(f"FinalGap = {fgap}") 
      data_modifying = (f" Modifying time = {end_algo - start_algo}")
      print(data_modifying)  
      newdata = from_networkx(newgraph)
      print(newdata)
      data.edge_index = torch.cat([newdata.edge_index])  
      return data,fgap,data_modifying

def proxyaddmax(args,data, nxgraph, max_iterations):
        print("Adding edges to maximize the gap...")
        start_algo = time.time()
        newgraph = process_and_update_edges(nxgraph, rank_by_proxy_add, "proxyaddmax",max_iter=max_iterations,updating_period=1)
        newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
        end_algo = time.time()
        fgap,_, _, _ = spectral_gap(newgraph)
        print()
        print(f"FinalGap = {fgap}") 
        data_modifying = (f" Modifying time = {end_algo - start_algo}")
        print(data_modifying)  
        newdata = from_networkx(newgraph)
        print(newdata)
        data.edge_index = torch.cat([newdata.edge_index])  
        return data,fgap,data_modifying


def proxyaddmin(args,data, nxgraph, max_iterations):
        print("Adding edges to minimize the gap...")
        start_algo = time.time()
        newgraph = min_and_update_edges(nxgraph, rank_by_proxy_add_min, "proxyaddmin",max_iter=max_iterations,updating_period=1)
        newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
        end_algo = time.time()
        fgap,_, _, _ = spectral_gap(newgraph)
        print()
        print(f"FinalGap = {fgap}") 
        data_modifying = (f" Modifying time = {end_algo - start_algo}")
        print(data_modifying)  
        newdata = from_networkx(newgraph)
        print(newdata)
        data.edge_index = torch.cat([newdata.edge_index])  
        return data,fgap,data_modifying


def fosr(data, max_iterations):
      print("Adding edges using FoSR...")
      start_algo = time.time()
      for j in tqdm(range((max_iterations))):
        edge_index,edge_type,_,prod = edge_rewire(data.edge_index.numpy(), num_iterations=1)      
        data.edge_index = torch.tensor(edge_index)
      data.edge_index = torch.cat([data.edge_index])
      end_algo = time.time()
      data_modifying = (f" Modifying time = {end_algo - start_algo}")
      newgraph = to_networkx(data, to_undirected=True)
      fgap,_, _, _ = spectral_gap(newgraph)
      return data, fgap, data_modifying

def sdrf(data, max_iterations,removal_bound,tau):
          #print("Rewiring using SDRF...")
          start_algo = time.time()
          Newdatapyg = sdrf(data,max_iterations,removal_bound,tau)
          end_algo = time.time()
          data_modifying = (f" Modifying time = {end_algo - start_algo}")
          newgraph = to_networkx(Newdatapyg, to_undirected=True)
          fgap,_, _, _ = spectral_gap(newgraph)
          data = from_networkx(Newdatapyg)
          return data, fgap, data_modifying