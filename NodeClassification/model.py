import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATv2Conv
from torch.nn import Sequential as Seq, Linear, ReLU, Dropout


class GCN(torch.nn.Module):
  def __init__(self, num_features,num_classes,hidden_channels, num_layers):
    super().__init__()
    self.num_layers = num_layers
    self.convs = torch.nn.ModuleList()
    self.convs.append(GCNConv(num_features, hidden_channels))
    for _ in range(num_layers - 1):
      self.convs.append(GCNConv(hidden_channels, hidden_channels))
    self.conv_out = GCNConv(hidden_channels,num_classes)
  
  def forward(self, x, edge_index,p):
    for i in range(self.num_layers):
      x = self.convs[i](x, edge_index)
      x = x.relu()
      x = F.dropout(x, p=p, training=self.training)
    x = self.conv_out(x, edge_index)
    return x

class GATv2(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)

    def forward(self, x, edge_index,p):
        h = F.dropout(x, p=p, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=p, training=self.training)
        h = self.gat2(h, edge_index)
        return F.log_softmax(h, dim=1)

# class GATv2Model(torch.nn.Module):
#   def __init__(self, num_node_features, num_classes, hidden_channels, num_layers):
#     super(GATv2Model, self).__init__()

#     self.num_layers = num_layers
#     self.convs = torch.nn.ModuleList()
#     self.convs.append(GATv2Conv(num_node_features, hidden_channels, heads=8))
#     for _ in range(num_layers - 1):
#       self.convs.append(GATv2Conv(hidden_channels*8, hidden_channels, heads=8))
#     self.conv_out = GATv2Conv(hidden_channels*8, num_classes, heads=1)

#   def forward(self, data):
#     x, edge_index = data.x, data.edge_index

#     for i in range(self.num_layers):
#         x = self.convs[i](x, edge_index)
#         x = ReLU()(x)
#         x = Dropout(p=self.p)(x, training=self.training)
#     x = self.conv_out(x, edge_index)

#     return x

  # def forward(self, data):
  #   x, edge_index = data.x, data.edge_index

  #   for i in range(self.num_layers):
  #     x = self.convs[i](x, edge_index)
  #     x = ReLU()(x)
  #     x = Dropout(p=p)(x, training=self.training)
  #   x = self.conv_out(x, edge_index)

  #   return x
