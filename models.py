import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GATConv, NNConv,GatedGraphConv,MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU


class Encoder(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(Encoder, self).__init__()
    self.w = nn.Linear(input_dim + hidden_dim, hidden_dim)
    # self.relu=nn.ReLU()
  def forward(self, x, h):
    z = self.w(torch.cat([x, h], dim=1))
    return z

class GAT(nn.Module):
  def __init__(self, hidden_dim, n_layers):
    super(GAT, self).__init__()
    self.n_layers = n_layers
    self.relu = nn.ReLU()
    self.GATConv = nn.ModuleList(
        [GATConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(n_layers)]
    )

  def forward(self, z, edge_index):
    out = z
    for conv in self.GATConv:
      out = self.relu(conv(out,edge_index))
    out = self.GATConv[-1](out, edge_index)
    return out

class Decoder(nn.Module):
  def __init__(self, hidden_dim):
    super(Decoder, self).__init__()

    self.w = nn.Linear(2*hidden_dim, hidden_dim)
    self.w1 = nn.Linear(hidden_dim, hidden_dim)
    self.bs = nn.BatchNorm1d(hidden_dim)
    self.relu = nn.ReLU()
    self.head = nn.Linear(hidden_dim, 1)
    self.sig = torch.nn.Sigmoid()
  def forward(self, h, z):
    out = self.w(torch.cat([h, z], dim=1)) 
    out = self.relu(out)
    out = self.relu(self.w1(out))
    y = self.head(out)
    return self.sig(y)


class Termination(nn.Module):
  def __init__(self, hidden_dim):
    super(Termination, self).__init__()
    self.w = nn.Linear(hidden_dim, 1)
    self.sig = torch.nn.Sigmoid()
  def forward(self, h):
    h_bar = torch.mean(h, dim=0)
    t = self.sig(self.w(h_bar)) 
    return t

class NEGA(nn.Module):
  def __init__(self, input_dim, hidden_dim, n_layers):
    super(NEGA, self).__init__()
    self.encoder = Encoder(input_dim, hidden_dim)
    self.processor_net = GAT(hidden_dim, n_layers)
    self.decoder = Decoder(hidden_dim)
    self.termination_net = Termination(hidden_dim)
  
  def forward(self, x, h, edge_index):
    z = self.encoder(x,h)
    h = self.processor_net(z, edge_index)
    out = self.decoder(h,z)
    t = self.termination_net(h)
    return out, t, h
