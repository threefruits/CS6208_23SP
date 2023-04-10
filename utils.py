import torch
import random
import numpy as np
import time
import os
import networkx as nx 
import numpy as np
from torch_geometric.utils import from_networkx
import torch 
import math
from matplotlib import pyplot as plt
from collections import deque



def save_model(model, model_name):
  # model_name = "{0}_average_accu_{1:.2f}_last_accuracy_{2:.2f}.pt".format(
  #   graph_type,
  #   n_nodes,
  # )
  if not os.path.isdir('./checkpoints'):
    os.mkdir('./checkpoints')
  
  path_to_save = os.path.join('./checkpoints',model_name)
  torch.save(model.state_dict(), path_to_save)


def load_model(model, path):
  model.load_state_dict(torch.load(path))
  return model

def generate_ladder_graph(n_nodes):
  graph = nx.ladder_graph(n_nodes)
  return graph

def generate_grid_graph(n_nodes):
  n_nodes_row = int ( (n_nodes) ** 0.5) + 1
  n_nodes_col = n_nodes_row 
  graph = nx.grid_2d_graph(n_nodes_row, n_nodes_col)
  return graph
  
def generate_trees_graph(n_nodes):
  graph = nx.random_tree(n_nodes)

  return graph

def generate_erdos_renyi_graph(n_nodes):
  p = min(np.log2(n_nodes)  / n_nodes, 0.5)
  graph = nx.erdos_renyi_graph(n_nodes, p)
  return graph

def generate_barbasi_albert_graph(n_nodes):
  n_attached_nodes = 4
  graph = nx.barabasi_albert_graph(n_nodes, n_attached_nodes, seed = np.random)
  return graph

def generate_4_community(n_nodes):
  n_nodes = n_nodes // 4
  graphs = [generate_erdos_renyi_graph(n_nodes) for _ in range(4)]
  
  graph = graphs[0]
  for i in range(1,len(graphs)):
    len_graph = len(graph.nodes)
    len_next_graph = len(graphs[i].nodes)
    graph = nx.union(graph, graphs[i], rename=('G', 'H'))
    
    G_nodes = ['G'+str(i) for i in range(len_graph)]
    H_nodes = ['H'+str(i) for i in range(len_next_graph)]
    size = len_graph * len_next_graph
    number_of_edges = np.sum(np.random.uniform(size=size) <= .01)
    g_nodes_to_connect = np.random.choice(G_nodes, replace=True, size=number_of_edges)
    h_nodes_to_connect = np.random.choice(H_nodes, replace=True, size=number_of_edges)
    edges = list(zip(g_nodes_to_connect, h_nodes_to_connect))
    graph.add_edges_from(edges)
  return graph

def init_e_weights(graph):
  n_edges = len(graph.edges)
  weights = np.random.uniform(low=0.2, high=1.0, size=n_edges)
  edge_to_weight = {edge: weights[i] for i, edge in enumerate(graph.edges())}
  nx.set_edge_attributes(graph, edge_to_weight, 'weight')

def init_n_weights(graph):
  n_nodes = len(graph.nodes)
  weights = np.random.uniform(low=0.2, high=1.0, size=n_nodes)
  node_to_weight = {node: weights[i] for i, node in enumerate(graph.nodes())}
  nx.set_node_attributes(graph, node_to_weight, 'nodes')


def breadth_first_search(graph, root=0):
  E = nx.to_numpy_matrix(graph)
  E=np.array(E)
  
  nb_nodes = graph.number_of_nodes()
  x = np.zeros((nb_nodes))
  x[root] = 1

  history = [x.copy()]

  queue = deque()
  queue.append(root)
  memory = set()
  terminate=False
  while len(queue) > 0 and np.sum(x) < len(x):
    second_queue = deque()
    while len(queue) > 0 and np.sum(x) < len(x):
      cur = queue.popleft()
      #print("cur",cur)
      memory.add(cur)
      neighbours = np.where(E[cur] > 0)[0]
      #print(E[cur])
      for n in neighbours:
        #print("n",n)
        if n not in memory:
          #print("added")
          second_queue.append(n)
          x[n] = 1
    if (x == history[-1]).all():
      terminate = True
      break
    history.append(x.copy())
    queue = second_queue
    if terminate:
      break
  return np.asarray(history)

def generate_graphs_and_states(n_sample, n_nodes, graph_type):
  targets = []
  graphs = []
  for i in range(n_sample):
    if graph_type == 'ladder':
      graph = generate_ladder_graph(n_nodes)
    elif graph_type == 'grid':
      graph = generate_grid_graph(n_nodes)
    elif graph_type == 'trees':
      graph = generate_trees_graph(n_nodes)
    elif graph_type == 'erdos_renyi':
      graph = generate_erdos_renyi_graph(n_nodes)
    elif graph_type == 'barabasi_albert':
      graph = generate_barbasi_albert_graph(n_nodes)
    elif graph_type == '4-community':
      graph = generate_4_community(n_nodes)

    init_e_weights(graph)
    init_n_weights(graph)

    n = len(graph.nodes)
    root = np.random.randint(0, n)


    states = breadth_first_search(graph,root)
    graph = from_networkx(graph)
    graph.x = torch.zeros(n)
    
    graph.x[root] = 1
    graphs.append(graph)
    targets.append(torch.tensor(states).float())
  return graphs, targets



    

if __name__ == '__main__':
  graph = generate_erdos_renyi_graph(20)
  E = nx.to_numpy_matrix(graph)
  E=np.array(E)

  hist = breadth_first_search(graph)
  for arr in hist:
    print(arr)
  
  print(E)


  # graph,states = generate_graphs_and_states(5, 20, 'erdos_renyi')
  # # E = nx.to_numpy_matrix(graph)
  
  
  # print(graph,states)
