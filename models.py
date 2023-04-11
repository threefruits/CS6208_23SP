import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GAT(nn.Module):
  def __init__(self, hidden_dim, n_layers):
    super(GAT, self).__init__()
    self.relu = nn.ReLU()
    self.GATConv = nn.ModuleList(
        [GATConv(in_channels=hidden_dim, out_channels=hidden_dim) for i in range(n_layers)]
    )

  def forward(self, z, edge_index):
    x = z
    for conv in self.GATConv:
      x = self.relu(conv(x,edge_index))
    out = self.GATConv[-1](x, edge_index)
    return out

class T_Net(nn.Module):
  def __init__(self, hidden_dim):
    super(T_Net, self).__init__()
    self.w = nn.Linear(hidden_dim, 1)
    self.sigmoid = torch.nn.Sigmoid()
  def forward(self, h):
    h = torch.mean(h, dim=0)
    t = self.sigmoid(self.w(h)) 
    return t
  
class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(Encoder, self).__init__()
    self.w = nn.Linear(input_dim + hidden_dim, hidden_dim)
  def forward(self, x, h):
    z = self.w(torch.cat([x, h], dim=1))
    return z

class Decoder(nn.Module):
  def __init__(self, hidden_dim):
    super(Decoder, self).__init__()

    self.l1 = nn.Linear(2*hidden_dim, hidden_dim)
    self.l2 = nn.Linear(hidden_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.out = nn.Linear(hidden_dim, 1)
    self.sig = torch.nn.Sigmoid()
  def forward(self, h, z):
    out = self.l1(torch.cat([h, z], dim=1)) 
    out = self.relu(out)
    out = self.relu(self.l2(out))
    y = self.out(out)
    return self.sig(y)

class NEGA(nn.Module):
  def __init__(self, input_dim, hidden_dim, n_layers):
    super(NEGA, self).__init__()
    self.encoder = Encoder(input_dim, hidden_dim)
    self.decoder = Decoder(hidden_dim)
    self.processor_net = GAT(hidden_dim, n_layers)
    self.termination_net = T_Net(hidden_dim)
  
  def forward(self, x, h, edge_index):
    z = self.encoder(x,h)
    h = self.processor_net(z, edge_index)
    out = self.decoder(h,z)
    terminate = self.termination_net(h)
    return out, terminate, h
