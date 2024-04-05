import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear


class GCN(torch.nn.Module):
  def __init__(self, num_features,num_classes,hidden_channels, num_layers,p=0.5):
    super().__init__()
    self.num_layers = num_layers
    self.convs = torch.nn.ModuleList()
    self.convs.append(GCNConv(num_features, hidden_channels))
    for _ in range(num_layers - 1):
      self.convs.append(GCNConv(hidden_channels, hidden_channels))
    self.conv_out = GCNConv(hidden_channels,num_classes)
  
  def forward(self, x, edge_index):
    for i in range(self.num_layers):
      x = self.convs[i](x, edge_index)
      x = x.relu()
      x = F.dropout(x, p=p, training=self.training)
    x = self.conv_out(x, edge_index)
    return x