import torch
import numpy as np
import torch.nn.functional as F
import time
from models import NEGA
import torch
import torch.optim as optim
import torch.nn as nn
from utils import save_model, load_model, generate_graphs_and_states
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--graph_type", type=str, default='barabasi_albert', help="graph_type")
parser.add_argument("--input_dim", type=int, default=1, help="input_dim")
parser.add_argument("--hidden_dim", type=int, default=32, help="hidden_dim")
parser.add_argument("--lr", type=float, default=0.0005, help="lr")
parser.add_argument("--layers", type=int, default=1, help="layers")
parser.add_argument("--epochs", type=int, default=20, help="epochs")
parser.add_argument("--n_samples_train", type=int, default=100, help="n_samples_train")
parser.add_argument("--n_nodes_train", type=int, default=20, help="n_nodes_train")
parser.add_argument("--n_samples_val", type=int, default=5, help="n_samples_val")
parser.add_argument("--n_nodes_val", type=int, default=20, help="n_nodes_val")
parser.add_argument("--n_samples_test", type=int, default=5, help="n_samples_test")
parser.add_argument("--n_nodes_test", type=int, default=100, help="n_nodes_test")

args = parser.parse_args()


if __name__ == '__main__':
    print('Preparing data....')
    start_time = time.time()
    train_graphs, train_states = generate_graphs_and_states(args.n_samples_train, args.n_nodes_train ,args.graph_type)
    val_graphs, val_states = generate_graphs_and_states(args.n_samples_val, args.n_nodes_val ,args.graph_type)
    test_graphs, test_states = generate_graphs_and_states(args.n_samples_test, args.n_nodes_test ,args.graph_type)
    end_time = time.time()
    print(f'Prepared in {end_time - start_time:.2f} seconds')
    print()

    if torch.cuda.is_available():
        print('use gpu')
        device = torch.device("cuda")
    else:
        print('use cpu')
        device = torch.device("cpu")

    model = NEGA(args.input_dim, args.hidden_dim, args.layers).to(device)
    print(model)
    model_path = './checkpoints/model.pt'
    model = load_model(model,model_path)
    

    # test(model, test_graphs, test_states)

    t_accus = []
    model.eval()
    for graph_no, (graph, states) in enumerate(zip(test_graphs,test_states)):
        x, edge_index = graph.x.to(device), graph.edge_index.to(device)
        states = states.to(device)
        N = x.shape[0]
        h = torch.zeros(N, args.hidden_dim).to(device)
        v = len(x)
        x=x.unsqueeze(dim=1)
        accus = []
        for i in range(v):
            out,t, h = model(x, h, edge_index)
            stop = 0
            if len(states)-1 == i:
                stop = 1
            x = (out > .5).to(int)
            target= states[i]
            out =( out > .5).to(int)
            out = out.squeeze(1)
            accu = torch.sum(out == target) / len(out)
            accus.append(accu.item())

            
            if stop ==1:
                break
            if t.item() > .5:
                    break

        a = np.mean(accus)
        l = accus[-1]
        print(f'Graph {graph_no} | Average Accuracy {a:.2f} | Last Accuracy {l:.2f}')
        t_accus.append(a)

            

    all_a = np.mean(t_accus)
    print(f'Average Accuracy Across All Graphs {all_a:.2f}')
    # np.mean(t_accus), t_accus[-1]

    
