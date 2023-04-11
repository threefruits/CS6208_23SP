import torch
import numpy as np
import networkx as nx 
import numpy as np
from torch_geometric.utils import from_networkx
import torch 
from collections import deque

def generate_ladder(n_nodes):
  graph = nx.ladder_graph(n_nodes)
  return graph

def generate_grid(n_nodes):
  n_nodes_row = int ( (n_nodes) ** 0.5) + 1
  n_nodes_col = n_nodes_row 
  graph = nx.grid_2d_graph(n_nodes_row, n_nodes_col)
  return graph
  
def generate_trees(n_nodes):
  graph = nx.random_tree(n_nodes)
  return graph

def generate_erdos_renyi(n_nodes):
  p = min(np.log2(n_nodes)  / n_nodes, 0.5)
  graph = nx.erdos_renyi_graph(n_nodes, p)
  return graph

def generate_barbasi_albert(n_nodes):
  n_attached_nodes = 4
  graph = nx.barabasi_albert_graph(n_nodes, n_attached_nodes, seed = np.random)
  return graph

def generate_4_community(n_nodes):
  # divide the total number of nodes by 4, assuming that we want to generate 4 communities
  n_nodes = n_nodes // 4
  graphs = [generate_erdos_renyi(n_nodes) for _ in range(4)]
  # combine the first graph with the other three graphs using union, and add edges between the nodes of different graphs
  graph = graphs[0]
  for i in range(1,len(graphs)):
    len_graph = len(graph.nodes)
    len_next_graph = len(graphs[i].nodes)
    graph = nx.union(graph, graphs[i], rename=('G', 'H'))
    # create lists of nodes from the first and second graphs to connect
    G_nodes = ['G'+str(i) for i in range(len_graph)]
    H_nodes = ['H'+str(i) for i in range(len_next_graph)]
    # determine the number of edges to add between the two sets of nodes, based on a probability of 0.01
    size = len_graph * len_next_graph
    number_of_edges = np.sum(np.random.uniform(size=size) <= .01)
    # randomly select nodes to connect from each set of nodes, and create edges between them
    g_nodes_to_connect = np.random.choice(G_nodes, replace=True, size=number_of_edges)
    h_nodes_to_connect = np.random.choice(H_nodes, replace=True, size=number_of_edges)
    edges = list(zip(g_nodes_to_connect, h_nodes_to_connect))
    graph.add_edges_from(edges)
  return graph

def breadth_first_search(graph, root=0):
  A = nx.to_numpy_matrix(graph)
  A = np.array(A)
  nb_nodes = graph.number_of_nodes()
  x = np.zeros((nb_nodes))
  # Set the root node to 1 in the x vector to start the search from the root node
  x[root] = 1
  history = [x.copy()]
  q = deque()
  q.append(root)
  memory = set()
  terminate = False
  while len(q) > 0 and np.sum(x) < len(x):
    second_queue = deque()
    # Loop through all nodes in the current level of the search
    while len(q) > 0 and np.sum(x) < len(x):
      cur = q.popleft()
      memory.add(cur)
      neighbours = np.where(A[cur] > 0)[0]
      for n in neighbours:
        if n not in memory:
          second_queue.append(n)
          x[n] = 1
    # If all the nodes in the current level have been visited, add the current x vector to the history list
    if (x == history[-1]).all():
      terminate = True
      break
    history.append(x.copy())
    q = second_queue
  # Return the history list as a NumPy array
  return np.asarray(history)

def generate_graphs_and_states(n_sample, n_nodes, graph_type):
  states = []
  graphs = []
  for i in range(n_sample):
    if graph_type == 'ladder':
      graph = generate_ladder(n_nodes)
    elif graph_type == 'grid':
      graph = generate_grid(n_nodes)
    elif graph_type == 'trees':
      graph = generate_trees(n_nodes)
    elif graph_type == 'erdos_renyi':
      graph = generate_erdos_renyi(n_nodes)
    elif graph_type == 'barabasi_albert':
      graph = generate_barbasi_albert(n_nodes)
    elif graph_type == '4-community':
      graph = generate_4_community(n_nodes)
    n = len(graph.nodes)
    root = np.random.randint(0, n)
    s = breadth_first_search(graph,root)
    graph = from_networkx(graph)
    graph.x = torch.zeros(n)
    graph.x[root] = 1
    graphs.append(graph)
    states.append(torch.tensor(s).float())
  return graphs, states